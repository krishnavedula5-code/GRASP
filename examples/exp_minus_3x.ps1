$output_dir = "outputs/exp_run"
Remove-Item -Recurse -Force $output_dir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $output_dir | Out-Null

$args = @(
  "-m", "numerical_lab.cli",
  "--expr", "exp(x) - 3*x",
  "--mode", "compare",
  "--a", "0", "--b", "2",
  "--x0", "0.5", "--x1", "1.5",
  "--numerical-derivative",
  "--export-report", "$output_dir/report.md",
  "--export-json-dir", "$output_dir/traces"
)

python @args