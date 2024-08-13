#!/usr/bin/env python3
import json

researchers = open("results.json", "r")
researchers = json.load(researchers)

count = 0
for name in researchers.keys():
    if len(researchers[name]) < 800:
        print(name)
        count += 1
print(count)
