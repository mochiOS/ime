use strict;
use warnings;
use utf8;

use Encode qw(decode encode);
use Getopt::Long qw(GetOptions);
use IPC::Open2;

binmode STDIN, ':encoding(UTF-8)';
binmode STDOUT, ':encoding(UTF-8)';
binmode STDERR, ':encoding(UTF-8)';

my $input;
my $output;

my $engine = 'target/debug/candidates';
my $dictionary = 'vendor/ja/ja.mime';
my $reading_script = 'training/reading.py';

my $limit = 32;

GetOptions(
	'input=s' => \$input,
	'output=s' => \$output,
	'engine=s' => \$engine,
	'dictionary=s' => \$dictionary,
	'reading-script=s' => \$reading_script,
	'limit=i' => \$limit,
) or die "invalid arguments\n";

die "--input is required\n"
	unless defined $input;

die "--output is required\n"
	unless defined $output;

open my $input_file, '<:encoding(UTF-8)', $input
	or die "cannot open $input: $!\n";

open my $output_file, '>:encoding(UTF-8)', $output
	or die "cannot open $output: $!\n";

my ($reading_out, $reading_in);

my $reading_pid = open2(
	$reading_out,
	$reading_in,
	'python',
	$reading_script,
);

binmode $reading_in, ':encoding(UTF-8)';
binmode $reading_out, ':encoding(UTF-8)';

while (my $sentence = <$input_file>) {
	chomp $sentence;
	$sentence =~ s/\r$//;

	next if $sentence eq '';

	print {$reading_in} $sentence, "\n";

	my $reading = <$reading_out>;

	die "reading process terminated unexpectedly\n"
		unless defined $reading;

	chomp $reading;
	$reading =~ s/\r$//;

	next if $reading eq '';

	my @candidates = generate_candidates(
		$engine,
		$dictionary,
		$reading,
		$limit,
	);

	print_example(
		$output_file,
		1,
		$sentence,
		$reading,
	);

	my %seen;
	$seen{$sentence} = 1;

	for my $candidate (@candidates) {
		next if $candidate eq '';
		next if $seen{$candidate}++;

		print_example(
			$output_file,
			0,
			$candidate,
			$reading,
		);
	}
}

close $reading_in;
waitpid $reading_pid, 0;

close $input_file;
close $output_file;

sub generate_candidates {
	my (
		$engine,
		$dictionary,
		$reading,
		$limit,
	) = @_;

	my @command = (
		$engine,
		'--dictionary',
		$dictionary,
		'--text',
		$reading,
		'--limit',
		$limit,
	);

	open my $pipe, '-|', @command
		or die "cannot execute $engine: $!\n";

	binmode $pipe, ':encoding(UTF-8)';

	my @candidates;

	while (my $candidate = <$pipe>) {
		chomp $candidate;
		$candidate =~ s/\r$//;

		push @candidates, $candidate
			if $candidate ne '';
	}

	close $pipe;

	return @candidates;
}

sub print_example {
	my (
		$file,
		$label,
		$text,
		$reading,
	) = @_;

	$text = escape_field($text);
	$reading = escape_field($reading);

	print {$file}
		$label,
		"\t",
		$reading,
		"\t",
		$text,
		"\n";
}

sub escape_field {
	my ($text) = @_;

	$text =~ s/\\/\\\\/g;
	$text =~ s/\t/\\t/g;
	$text =~ s/\r/\\r/g;
	$text =~ s/\n/\\n/g;

	return $text;
}

# AI時代、Python一強なのにもかかわらずPerlを使う逆張りマン
#
#       - tas0dev