# Fixtures

- `jfk.wav`: 11 s, 16 kHz mono PCM, JFK inaugural address excerpt.
  Source: https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav
  SHA-256: `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`
- `test-config.toml`: config used by `scripts/test.sh`; listens on 127.0.0.1:18797 so it never collides with a real install on 8797.
- `jfk.txt`: reference text for `jfk.wav`, part of the parity corpus.
- `librispeech/`: ten LibriSpeech test-clean utterances (`<utt-id>.flac`) and `references.txt` (`<utt-id> <TRANSCRIPT>`).
  Source: https://www.openslr.org/resources/12/test-clean.tar.gz (LibriSpeech, V. Panayotov et al., ICASSP 2015). License: CC BY 4.0.
  Selection rule: sort speaker ids numerically, take the first 10 speakers; for each, the first utterance (sorted by chapter, then id) whose duration is between 5 and 20 s.
  Ids: 61-70968-0005, 121-121726-0000, 237-126133-0000, 260-123286-0000, 672-122797-0001, 908-157963-0000, 1089-134686-0000, 1188-133604-0000, 1221-135766-0000, 1284-1180-0000.
- `vocabulary.wav`, `vocabulary.txt`: synthetic speech, no personal audio. Made once with the local Kokoro server (`POST http://127.0.0.1:8880/v1/audio/speech`, model `kokoro`, voice `af_heart`, `response_format` wav), then `ffmpeg -ar 16000 -ac 1 -c:a pcm_s16le`. 6.5 s. The text is in `vocabulary.txt`; it carries the four terms of the `default` profile in `test-config.toml` (Oluwaseun, Ghidra, Yggdrasil, Zstandard). Without the profile whisper mishears at least some of them.
