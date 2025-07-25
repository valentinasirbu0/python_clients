# C:\Users\valentina.sirbu\OneDrive - AMDARIS GROUP LIMITED\Desktop\Test_Project2\Test_Project\python-clients\scripts\asr\transcribe_file_offline.py

# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

import grpc
import riva.client
# Ensure these imports are correct based on your nvidia-riva-client version
from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline file transcription via Riva AI Services. \"Offline\" means that entire audio "
        "content of `--input-file` is sent in one request and then a transcript for whole file recieved in "
        "one response.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", required=True, type=Path, help="A path to a local file to transcribe.")
    parser = add_connection_argparse_parameters(parser)
    
    # FIX 1: Add speaker_diarization=True to include diarization-related arguments
    parser = add_asr_config_argparse_parameters(
        parser, 
        max_alternatives=True, 
        profanity_filter=True, 
        word_time_offsets=True,
        speaker_diarization=True # <-- ADD THIS LINE
    )

    # FIX 2: REMOVE or COMMENT OUT this entire block that manually adds --custom-configuration
    # This was causing the "conflicting option string" error because add_asr_config_argparse_parameters
    # likely already adds it.
    # parser.add_argument(
    #     "--custom-configuration",
    #     action='append',
    #     nargs='*',
    #     help="A key-value pair or pairs for custom configuration, e.g., --custom-configuration key=value."
    # )

    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    return args


def main() -> None:
    args = parse_args()
    auth = riva.client.Auth(args.ssl_cert, args.use_ssl, args.server, args.metadata)
    asr_service = riva.client.ASRService(auth)
    config = riva.client.RecognitionConfig(
        language_code=args.language_code,
        max_alternatives=args.max_alternatives,
        profanity_filter=args.profanity_filter,
        enable_automatic_punctuation=args.automatic_punctuation,
        verbatim_transcripts=not args.no_verbatim_transcripts,
        enable_word_time_offsets=args.word_time_offsets or args.speaker_diarization,
    )
    # Ensure these arguments are available via parse_args if you use word boosting
    riva.client.add_word_boosting_to_config(config, args.boosted_lm_words, args.boosted_lm_score)
    
    # FIX 3: Pass all required arguments to add_speaker_diarization_to_config
    # Based on the TypeError, your riva.client version expects 3 positional arguments here.
    riva.client.add_speaker_diarization_to_config(
        config, 
        args.speaker_diarization, 
        args.diarization_min_speakers, # <-- ADD THIS ARGUMENT
        args.diarization_max_speakers  # <-- ADD THIS ARGUMENT
    )
    
    # Ensure these arguments are available via parse_args if you use endpoint parameters
    riva.client.add_endpoint_parameters_to_config(
        config,
        args.start_history,
        args.start_threshold,
        args.stop_history,
        args.stop_history_eou,
        args.stop_threshold,
        args.stop_threshold_eou
    )
    
    # IMPORTANT: Keep this line commented out/removed unless your riva.client version
    # truly provides 'add_custom_configuration_to_config'
    # riva.client.add_custom_configuration_to_config(
    #     config,
    #     args.custom_configuration
    # )
    
    with args.input_file.open('rb') as fh:
        data = fh.read()
    try:
        riva.client.print_offline(response=asr_service.offline_recognize(data, config))
    except grpc.RpcError as e:
        print(f"Riva gRPC Error: {e.details()}")
    except Exception as e:
        print(f"An unexpected error occurred during Riva API call: {e}")


if __name__ == "__main__":
    main()
