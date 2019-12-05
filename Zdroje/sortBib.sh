#!bin/bash

sort bib.txt | tr -s '\n' | sed '1d' | sed = | sed 'n;G' | sed -E 's/^[0-9]+$/\[&\]/g' | sed -e '/\]$/N' -e 's/\n/\ /' > sortedBib.txt