# Basic Rules

## Data format
Image: N C H W (same as pytorch), uint8

Action (discrete): int64

Other float: float32

## Distributed
- Lock-free
- Copy-free

## State
Log node activities as state

Non-working load:
- wait_xxx
- copy_xxx

Working load:
- step
- reset
- process
- update
