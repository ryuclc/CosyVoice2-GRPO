import os 
import asyncio
import aiohttp
import re
import logging
logging.getLogger('asyncio').setLevel(logging.WARNING)


SERVER = os.getenv("WHISPER_SERVER", "http://172.16.46.216:8080")
SCORE_URL = f"{SERVER.rstrip('/')}/end_to_end"
HEALTH_URL = f"{SERVER.rstrip('/')}/health"

# export WHISPER_SERVER=http://172.16.46.216:8080


def remove_lang_tag(text: str) -> str:
    return re.sub(r'^<\|[^|]+?\|>', '', text)


async def process_audio_sample(sampling_rate, speech_tokens_str: str, expected_answer: str) -> float:
    """
    Process a single sample by calling the combined /end_to_end endpoint,
    which decodes speech tokens and transcribes the resulting audio.
    The service returns (among others) the transcribed text, WER, and reward.
    """
    payload = {
        "speech_tokens_str": speech_tokens_str,
        "sample_rate": sampling_rate,
        "expected_text": remove_lang_tag(expected_answer),
    }
    transcribed_text = ""
    language = ""
    cer = 0.0
    nll = 0.0
    reward = 0.0
    cer_reward = 0.0
    nll_reward = 0.0
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                SCORE_URL, json=payload, timeout=300
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    transcribed_text = result.get("transcribed_text", "")
                    language = result.get("language", "")
                    cer = result.get("cer", 0.0)
                    nll = result.get("nll", 0.0)
                    reward = result.get("reward", 0.0)
                    cer_reward = result.get("cer_reward", 0.0)
                    nll_reward = result.get("nll_reward", 0.0)
                else:
                    error_text = await response.text()
                    print(
                        f"Error in combined endpoint: {response.status} - {error_text}"
                    )
    except Exception as e:
        print(f"Exception in combined endpoint request: {e}")

    # print(
    #     "-" * 20,
    #     f"\nExpected Answer:\n{expected_answer}",
    #     f"\nTranscribed:\n{transcribed_text}",
    #     f"\nCER: {cer}, Reward: {reward}",
    #     f"\nCER Reward: {result.get('cer_reward', None)}, NLL Reward: {result.get('nll_reward', None)}",
    # )
    # return reward tuple, first is CER reward, second is NLL reward, third is harmonic mean reward
    return cer_reward, nll_reward, reward, cer, nll, language



async def wer_reward_func_async(
    sampling_rate, speech_tokens_list: list[str], answers: list[str]
) -> list[float]:
    """
    Async version of the reward function that processes all samples in
    parallel using the combined endpoint.
    """
    tasks = [
        process_audio_sample(sampling_rate, speech_tokens, answer)
        for speech_tokens, answer in zip(speech_tokens_list, answers)
    ]
    rewards = await asyncio.gather(*tasks)
    return rewards


def wer_reward_func(sampling_rate, completions, answer, **kwargs) -> list[float]:
    """
    Synchronous interface for the async reward function.
    Processes all transcription requests in parallel using the combined endpoint.
    Expects the completions to be a list where each element is a list/dict
    that contains the speech token string in completion[0]['content'].
    """
    speech_tokens_list = completions
    return asyncio.run(wer_reward_func_async(sampling_rate, speech_tokens_list, answer))

