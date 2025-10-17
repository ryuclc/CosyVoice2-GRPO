# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
import numpy as np
import threading
# import time
from torch.nn import functional as F
from contextlib import nullcontext
# import uuid
from cosyvoice.utils.common import fade_in_out

from torch.nn.utils.rnn import pad_sequence, unpad_sequence


class GRPOModel:

    def __init__(self, num_generations, llm, flow, hift, device):
        self.device = device
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.num_generations = num_generations


    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding):
        # with self.llm_context:
        tts_speech_token_dict = []
        for i in self.llm.inference(text=text.to(self.device),
                                    text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                    prompt_text=prompt_text.to(self.device),
                                    prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                    prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                    prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                    embedding=llm_embedding.to(self.device)):
            tts_speech_token_dict.append(i)
        return tts_speech_token_dict

    def token2wav(self, token, prompt_token, prompt_feat, embedding, stream=False, finalize=True):
        tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_token=prompt_token.to(self.device),
                                            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_feat=prompt_feat.to(self.device),
                                            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                            embedding=embedding.to(self.device),
                                            streaming=stream,
                                            finalize=finalize)

        hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
    
        tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)

        return tts_speech.cpu().squeeze().tolist()


    def generate_grpo(self, text_token, text_token_len, speech_token, speech_token_len, embedding, speech_feat, speech_feat_len, zero_shot=False, cross_lingual=False):
        # input (B * *)
        # due to cosyvoice only surpport batch_size=1 inference
        # unpad_sequence to LIST[]
        # loop inference
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        speech_feat = unpad_sequence(speech_feat, speech_feat_len.cpu(), batch_first=True)

        generations = []
        completion_ids = []
        
        for batch_num, (text_token_one, text_token_len_one, speech_token_one, speech_token_len_one, speech_feat_one, speech_feat_len_one, embedding_one) in enumerate(
            zip(text_token, text_token_len, speech_token, speech_token_len, speech_feat, speech_feat_len, embedding)
        ):
            # cosyvoice2, force speech_feat % speech_token = 2
            token_len = min(int(speech_feat_one.shape[0] / 2), speech_token_one.shape[0])
            speech_feat_one, speech_feat_len_one = speech_feat_one[:2 * token_len], 2 * token_len
            speech_token_one, speech_token_len_one = speech_token_one[:token_len], token_len

            text = text_token_one.unsqueeze(0)
            # during grpo, set embedding to 0 !!!
            flow_embedding = torch.zeros(1, 192)
            llm_embedding=torch.zeros(0, 192)
            prompt_text=torch.zeros(1, 0, dtype=torch.int32)
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32)
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32)
            prompt_speech_feat=torch.zeros(1, 0, 80)

            if zero_shot is True:
                flow_embedding = embedding_one.unsqueeze(0)
                llm_embedding = embedding_one.unsqueeze(0)
                if cross_lingual is False:
                    prompt_text = text_token_one.unsqueeze(0)
                    llm_prompt_speech_token = speech_token_one.unsqueeze(0)
                    flow_prompt_speech_token = speech_token_one.unsqueeze(0)
                    prompt_speech_feat = speech_feat_one.unsqueeze(0)
                else:
                    flow_prompt_speech_token = speech_token_one.unsqueeze(0)
                    prompt_speech_feat = speech_feat_one.unsqueeze(0)

            for gen_num in range(self.num_generations):
                tts_speech_token_dict = self.llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding)

                this_tts_speech_token = torch.tensor(tts_speech_token_dict).unsqueeze(dim=0)

                tts_speech = self.token2wav(this_tts_speech_token, flow_prompt_speech_token, prompt_speech_feat, flow_embedding, stream=False, finalize=True)
                generations.append(tts_speech)
                completion_ids.append(tts_speech_token_dict)

        torch.cuda.empty_cache()
        return generations, completion_ids
            