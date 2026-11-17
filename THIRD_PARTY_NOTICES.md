# Sources and notices

- **ImageWAM**: https://github.com/yuyangalin/ImageWAM, upstream checkout `5d4a341ed20a95cdb08f0293f3d44778b9a9e05a`; local SIMPLE RGB-D and geometry modifications are included. The original MIT notice is preserved in `LICENSE`.
- **FLUX.2 source snapshot**: copied from the ImageWAM checkout's `third_party/flux2/src`; retain `third_party/flux2/LICENSE.md` (Apache-2.0). This snapshot was not an independent Git checkout; no separate commit is claimed. File hashes are in `RELEASE_MANIFEST.json`. Official source: https://github.com/black-forest-labs/flux2.
- **Depth Pro source snapshot**: https://github.com/apple/ml-depth-pro. Its original `LICENSE` and `ACKNOWLEDGEMENTS.md` are retained under `third_party/ml-depth-pro/`. Only code/documentation is included, not pretrained weights. No source commit was available in the local source archive; hashes identify the exact included files.

Model weights and the SIMPLE RGB-D dataset are separate downloads, and are not relicensed by the root ImageWAM code license. Preserve the notices in the vendored files when redistributing this repository.
