#!/bin/bash
make
./benchmark > logs/execution_log.txt
cat logs/execution_log.txt
