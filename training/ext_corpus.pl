use strict;
use warnings;
use utf8;

use Encode qw(decode);
use File::Find;
use Getopt::Long qw(GetOptions);
use IO::Uncompress::Gunzip qw($GunzipError);
use JSON::PP qw(decode_json);

binmode STDOUT, ':encoding(UTF-8)';
binmode STDERR, ':encoding(UTF-8)';

my $input_dir;
my $output;

GetOptions(
	'input-dir=s' => \$input_dir,
	'output=s' => \$output,
) or die "invalid arguments\n";

die "--input-dir is required\n"
	unless defined $input_dir;

die "--output is required\n"
	unless defined $output;

open my $out, '>:encoding(UTF-8)', $output
	or die "cannot open $output: $!\n";

my @files;

find(
	sub {
		return unless -f $_;
		return unless $_ =~ /\.jsonl\.gz$/;

		push @files, $File::Find::name;
	},
	$input_dir,
);

@files = sort @files;

die "no .jsonl.gz files found in $input_dir\n"
	unless @files;

my $document_count = 0;
my $sentence_count = 0;

for my $path (@files) {
	print STDERR "processing $path\n";

	my $gz = IO::Uncompress::Gunzip->new($path)
		or die "cannot open $path: $GunzipError\n";

	while (my $line = <$gz>) {
		my $decoded = decode('UTF-8', $line);

		my $row;

		eval {
			$row = decode_json($decoded);
		};

		next if $@;
		next unless ref $row eq 'HASH';

		my $body =
			$row->{text}
			// $row->{content}
			// $row->{body};

		next unless defined $body;
		next if $body eq '';

		$document_count++;

		for my $sentence (split_sentences($body)) {
			next unless is_usable_sentence($sentence);

			print {$out} $sentence, "\n";
			$sentence_count++;
		}
	}

	close $gz;

	print STDERR
		"documents=$document_count "
		. "sentences=$sentence_count\n";
}

close $out;

print STDERR "done\n";
print STDERR "documents: $document_count\n";
print STDERR "sentences: $sentence_count\n";

sub split_sentences {
	my ($text) = @_;

	$text =~ s/\r\n?/\n/g;
	$text =~ s/[ \t]+/ /g;

	my @sentences;

	for my $paragraph (split /\n+/, $text) {
		$paragraph =~ s/^\s+//;
		$paragraph =~ s/\s+$//;

		next if $paragraph eq '';

		my @parts = split /(?<=[。！？!?])/u, $paragraph;

		for my $sentence (@parts) {
			$sentence =~ s/^\s+//;
			$sentence =~ s/\s+$//;

			push @sentences, $sentence
				if $sentence ne '';
		}
	}

	return @sentences;
}

sub is_usable_sentence {
	my ($sentence) = @_;

	my $length = length($sentence);

	return 0 if $length < 5;
	return 0 if $length > 200;

	return 0 if $sentence =~ m{https?://}i;
	return 0 if $sentence =~ /\bwww\./i;

	my $japanese = () =
		$sentence =~ /[\x{3040}-\x{30ff}\x{3400}-\x{9fff}]/g;

	return 0 if $japanese < 3;

	return 1;
}