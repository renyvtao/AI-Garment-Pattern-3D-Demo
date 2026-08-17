# Motion asset notices

## CMU Graphics Lab Motion Capture Database

The raw ASF/AMC files used for the `standing_turn`, `wave`, and
`walk_in_place` actions are downloaded from the Carnegie Mellon University
Graphics Lab Motion Capture Database:

- https://mocap.cs.cmu.edu/
- Subject 69, trials 01 and 16
- Subject 141, trial 16

CMU states that the motions are free for all uses and may be included in
commercially sold products, but the motion data may not itself be resold,
including in converted form. Consequently the application treats converted
SMPL-X motion files as internal runtime assets and does not include them in a
user result download ZIP.

Requested acknowledgement:

> The data used in this project was obtained from mocap.cs.cmu.edu. The
> database was created with funding from NSF EIA-0196217.

Hand and toe channels in the CMU source can be noisy. The retargeting stage
uses body, shoulder, elbow, and wrist motion and keeps SMPL-X fingers neutral.
