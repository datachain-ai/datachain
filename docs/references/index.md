---
title: API Reference
---

# API Reference

DataChain's API is organized into several modules:

- [DataChain](./datachain.md) - Core chain operations and dataset management
- [Data Types](./data-types/index.md) - Supported data types and schema definitions
    - [File](./data-types/file.md) - File handling and storage operations
    - [TextFile](./data-types/textfile.md) - Text file
    - [ImageFile](./data-types/imagefile.md) - Image file
    - [VideoFile](./data-types/videofile.md) - Video file
    - [AudioFile](./data-types/audiofile.md) - Audio file
    - [TarVFile](./data-types/tarvfile.md) - Virtual file model for files extracted from tar archives
    - [ArrowRow](./data-types/arrowrow.md) - Working with Arrow-supported file
    - [BBox](./data-types/bbox.md) - Bounding box data type
    - [Pose](./data-types/pose.md) - Pose data type
    - [Segment](./data-types/segment.md) - Segment data type
- [UDF](./udf.md) - User-defined functions and transformations
- [Functions](./func.md) - Built-in functions for data manipulation and analysis
- [Torch](./torch.md) - PyTorch data loading utilities
- [Toolkit](./toolkit.md) - Functions for common DS/ML operations
