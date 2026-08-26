# Organizer welcome and research directions

- competition: `biohub-cell-tracking-during-development`
- source_url: <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716062>
- title: Welcome to the Biohub - Cell Tracking During Development Challenge
- author: Thibaut Goldsborough / Biohub organizer
- posted_at: 2026-06-30
- checked_at: 2026-08-26
- evidence_class: organizer guidance

## Summary

The organizers frame the task as robust cell detection, temporal association, division detection,
and lineage reconstruction in noisy, dense 3D+time zebrafish microscopy. The annotations are sparse:
training must not interpret every unlabeled location as background, while inference still has to track
all cells. Evaluation samples a random sparse subset of those cells.

The organizer points participants to the maintained
[`royerlab/kaggle-cell-tracking-competition`](https://github.com/royerlab/kaggle-cell-tracking-competition)
repository for data access, the baseline, visualization, CSV/GEFF conversion, and the metric. The
repository is the first implementation reference; the Kaggle overview and Rules remain the contract.

## Official clarifications in the thread

- The four public test clips are dummy notebook smoke-test inputs and may duplicate train clips.
- The hidden scoring test is much larger and does not overlap train.
- The organizer repository implements the official metric and now includes CSV-to-GEFF conversion.
- Public-test performance is therefore not validation evidence; use it only to verify notebook output.

## Referenced ecosystem

| Role | Resources | Use in this workspace |
| --- | --- | --- |
| Format and arrays | GEFF, Zarr, tracksdata | adopt for graph/image I/O |
| GPU image operations | CuPy, cuCIM | evaluate only after a CPU smoke path works |
| Visualization | napari, ndv | optional local inspection extra |
| Graph optimization | motile | candidate linker/lineage optimizer |
| Competitive tracking | ultrack, trackastra, byotrack, laptrack, CELLECT, ASCENT, OrganoidTracker | benchmark candidates; do not install all by default |
| External benchmarks | Cell Tracking Challenge | candidate external data and generalization evaluation |

Related organizer-comment references include the MPM cell-tracking paper (CVPR 2020) and ELEPHANT
(Nature Biotechnology 2022). They are research inputs, not competition rules.

## Actionable points

1. Use positive/sparse supervision; never manufacture negative labels from unlabeled cells.
2. Separate the public-test smoke path from embryo-disjoint local validation and hidden-LB evidence.
3. Start with the pinned organizer metric/I/O stack before comparing external trackers.
4. Record external dataset license, provenance, and Kaggle Rules compatibility before training.
5. Compare one tracker family at a time under the 12-hour offline Notebook constraint.

## Confidence and limits

Organizer statements are high-confidence guidance as checked on 2026-08-26. Package APIs and
third-party method quality are not guaranteed; versions must be pinned and validated locally.
