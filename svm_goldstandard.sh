#!/bin/sh

svm-scale  -l -1 -u 1 -s query_silver_all.min.out.range query_silver_all.min.out > query_silver_all.min.out.scaled
#svm-train -c 4 -t 0 -e 0.1 -m 800 -v 5 query_silver_all.min.out.scaled
svm-train -c 4 -t 0 -e 0.1 -m 800 query_silver_all.min.out.scaled
svm-predict query_silver_all.min.out.scaled.t query_silver_all.min.out.scaled.model query_silver_all.min.out.scaled.t.predict
