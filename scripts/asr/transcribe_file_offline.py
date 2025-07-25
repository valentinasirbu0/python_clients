# C:\Users\valentina.sirbu\OneDrive - AMDARIS GROUP LIMITED\Desktop\Test_Project2\Test_Project\python-clients\scripts\asr\transcribe_file_offline.py

# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

import grpc
import riva.client
from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline file transcription via Riva AI Services. \"Offline\" means that entire audio "
        "content of `--input-file` is sent in one request and then a transcript for whole file recieved in "
        "one response.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", required=True, type=Path, help="A path to a local file to transcribe.")
    parser = add_connection.argparse_parameters(parser)
    
    # Add ASR configuration parameters, which should include diarization arguments
    parser = add_asr_config.argparse_parameters(
        parser, 
        max_alternatives=True, 
        profanity_filter=True, 
        word_time_offsets=True 
    )

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
    riva.client.add_word_boosting_to_config(config, args.boosted_lm_words, args.boosted_lm_score)
    
    # This call remains unchanged from the previous 'final working code'.
    # args.speaker_diarization will now be implicitly available from add_asr_config_argparse_parameters,
    # and min/max speakers from our manual additions.
    riva.client.add_speaker_diarization_to_config(
        config, 
        args.speaker_diarization, 
        args.diarization_min_speakers, 
        args.diarization_max_speakers  
    )
    
    riva.client.add_endpoint_parameters_to_config(
        config,
        args.start_history,
        args.start_threshold,
        args.stop_history,
        args.stop_history_eou,
        args.stop_threshold,
        args.stop_threshold_eou
    )
    
    # The riva.client.add_custom_configuration_to_config call should remain commented out.
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
