# Path Functions

Functions for working with file paths and extracting path components.

`parent("d//e.txt")` returns `"d/"`, preserving the empty directory segment.
`File.parent` uses the same rule for nested paths.

::: datachain.func.path
