#!/usr/bin/perl
use strict;
use warnings;
use Data::Dumper;

#my $file = "test.txt";
#my $file = "/home/socr/b/rsd/gold_standard/query_db.out";
my $file = "/home/socr/b/rsd/gold_standard/query_silver_all.out";
open FILE, "<$file";
while (<FILE>) {
    chomp;

    my @tokes = split(/\t/);
    my $label = shift @tokes;
    print $label;
    shift @tokes;
    foreach my $toke (@tokes) {
    	my (undef,$score) = split(/:/,$toke);
    	print "\t" . $score;
    }
    print "\n";

}
close FILE;
