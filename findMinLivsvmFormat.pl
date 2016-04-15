#!/usr/bin/perl
use strict;
use warnings;
use Data::Dumper;

#my $file = "test.txt";
#my $file = "/home/socr/b/rsd/gold_standard/query_db.out";
#my $file = "/home/socr/b/rsd/gold_standard/query_silver_all.out";
my $file = "/Users/rsd/Dropbox/BabbittLabDropBox/gold_standard_dropbox/silver_standard_all_Apr15_2016/query_silver_all.out";
open FILE, "<$file";
my $min = 100000;
while (<FILE>) {
    chomp;

    my @tokes = split(/\t/);
    my $label = shift @tokes;
#    print $label;
    shift @tokes;
    foreach my $toke (@tokes) {
    	my (undef,$score) = split(/:/,$toke);
#	print $score . "\n";
    	if ($score ne "NA" && $score < $min) {
	    $min = $score;
	}
    }
}
close FILE;
print $min . "\n";
