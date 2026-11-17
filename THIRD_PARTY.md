# Third-party provenance

The root MIT license applies to project-owned code. External projects retain their own licenses. This repository fetches the upstream NPGC/RecBole source instead of bundling it. No original dataset or pretrained model is distributed.

| Component | Public source and pinned commit | Use and license |
|---|---|---|
| NPGC with its RecBole fork (RecBole 1.1.1) | https://github.com/zyouyang/RecSys2025_NonParamGC — `dd3c7f724037b40c04387b031b61fe8715a9e73c` | Framework, context backbones, official NPGC comparison. Upstream MIT notice is retained in `licenses/RECBOLE_MIT.txt`. |
| LibRerank PRM | https://github.com/LibRerank-Community/LibRerank — `7968d6d990268246ef01c8f53ea55724a48a42d1` | Reference for the project-local PyTorch PRM adapter. Preserve the MIT notice in `licenses/LIBRERANK_LICENSE.txt`. |
| QNN_alpha | https://github.com/salmon1802/QNN — `08c5db81e76b1a377708ce6e6164f3d7a3a2486d` | Reference for the RecBole QNN_alpha adapter patch. Apache 2.0 terms are in `licenses/QNN_APACHE_2.0.txt`. Consult the patch header for exact adaptation provenance. |
| Principled re-ranking | https://github.com/goldenxingxing/principled_reranking — `0aba140ce7cc71fb5cc9750f3811702edaa137ce` | Method reference for the local PyTorch loss implementation. No upstream source tree is redistributed. No license was recorded at the audited revision, so this project does not grant a license for that upstream code. |

The external patch files under `patches/` change only the pinned checkout: the dataset patch preserves an explicitly supplied label field, and the QNN_alpha patch adds the adapter used by the S1 experiment. `scripts/setup_third_party.py` applies them after checking the pinned revision. Existing dirty checkouts are not silently reset. The patches do not change the reported data split or reranking equations.

Local graph/reranking code under `src/min_path_mean`, feature/fusion code under `src/diagnostics`, and study orchestration under `src/studies` were extracted from the actual experiment paths. Shared candidate adapters under `src/baselines/dnr` are included solely because S1/S3 depend on them. No DNR model or DNR training/result bundle is included.

Dataset ownership and terms remain with the providers. Obtain the processed Yelp2018/Amazon-Books archives through the official NPGC data links and MovieLens-1M through GroupLens. Python dependencies retain their respective licenses. Public attribution in this document and the retained license notices identifies third parties, not the anonymous study authors.
