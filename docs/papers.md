# Reference list (verified)

Every entry below was checked against arXiv, a conference proceedings page,
or the official documentation. Entries marked "to read" still need to be
read in full before we cite specific claims from them.

## Detection models

1. **YOLOv10: Real-Time End-to-End Object Detection.**
   Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han,
   Guiguang Ding. arXiv:2405.14458, 2024. https://arxiv.org/abs/2405.14458
   *Use:* introduces NMS-free training with consistent dual assignments.
   Background for why removing NMS matters for latency and deployment,
   and for the NMS-free design that YOLO26 also adopts. (To read.)

2. **DETRs Beat YOLOs on Real-time Object Detection (RT-DETR).**
   Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang,
   Qingqing Dang, Yi Liu, Jie Chen. CVPR 2024, pp. 16965-16974.
   arXiv:2304.08069. https://arxiv.org/abs/2304.08069
   *Use:* shows that NMS affects the speed of YOLO detectors and proposes an
   end-to-end benchmark. Motivates reporting end-to-end latency, not
   model-only latency. (To read.)

3. **Ultralytics YOLO26 documentation.** Ultralytics, released January 2026.
   https://docs.ultralytics.com/models/yolo26
   *Use:* source of the vendor claims we test in H3 (NMS-free end-to-end
   design, DFL removal, "up to 43% faster CPU inference"). No peer-reviewed
   paper was found for YOLO26, so cite the documentation and repository
   (https://github.com/ultralytics/ultralytics) and label the claims as
   vendor-reported.

4. **Ultralytics YOLO11** is also cited as software (documentation and
   repository), not as a paper.

## Quantization and edge inference

5. **Quantization and Training of Neural Networks for Efficient
   Integer-Arithmetic-Only Inference.** Benoit Jacob, Skirmantas Kligys,
   Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam,
   Dmitry Kalenichenko. CVPR 2018, pp. 2704-2713. arXiv:1712.05877.
   *Use:* foundational 8-bit integer quantization scheme, with COCO
   detection results on CPUs. Basis for H1. (To read.)

6. **A White Paper on Neural Network Quantization.** Markus Nagel,
   Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko,
   Mart van Baalen, Tijmen Blankevoort. arXiv:2106.08295, 2021.
   https://arxiv.org/abs/2106.08295
   *Use:* practical guide to post-training quantization (PTQ) versus
   quantization-aware training (QAT). Justifies our PTQ-only scope and the
   calibration protocol. (To read.)

## Dataset and evaluation

7. **Microsoft COCO: Common Objects in Context.** Tsung-Yi Lin, Michael
   Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan,
   Piotr Dollar, C. Lawrence Zitnick. ECCV 2014, pp. 740-755.
   arXiv:1405.0312. https://arxiv.org/abs/1405.0312
   *Use:* dataset citation. Evaluation uses the official COCO bbox metrics
   through pycocotools.

## Reading order for Phase 1 (4-5 papers)

1 -> 2 -> 6 -> 5, then skim 7. Write 3-4 sentences per paper in your own
words in `docs/notes.md`: what problem, what method, what result, and how it
changes our experimental design.
