# AudioFile

`AudioFile` extends [`File`](file.md) and provides additional methods for working with audio files.

`AudioFile` instances are created when a `DataChain` is initialized [from storage](../datachain.md#datachain.lib.dc.storage.read_storage) with the `type="audio"` parameter:

```python
import datachain as dc

chain = dc.read_storage("s3://bucket-name/", type="audio")
```

There are additional models for working with audio files:

- `AudioFragment` - represents a fragment of an audio file.

These are virtual models that do not create physical files.
Instead, they are used to represent the data in the `AudioFile` these models are referring to.
If you need to save the data, you can use the `save` method of these models,
allowing you to save data locally or upload it to a storage service.

For a complete example of audio processing with DataChain, see
[Audio-to-Text with Whisper](https://github.com/datachain-ai/datachain/blob/main/examples/multimodal/audio-to-text.py) —
a speech recognition pipeline that uses `AudioFile`, `AudioFragment`, and `Audio`
to chunk audio files and transcribe them.

::: datachain.lib.file.AudioFile

::: datachain.lib.file.AudioFragment

::: datachain.lib.file.Audio
