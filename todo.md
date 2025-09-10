Maybe simulate network latency? (so error requests don't complete in 0 ms)
Refactor how you track and export faults, I don't like how it is right now (ugly)
Easy: Implement Queue discipline -  FCFS / LCFS / RSS / RR / Prio