$output_dir = "outputs/cosx_run"
Remove-Item -Recurse -Force $output_dir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $output_dir | Out-Null

$args = @(
  "-m", "numerical_lab.cli",
  "--expr", "cos(x) - x",
  "--mode", "compare",
  "--a", "0", "--b", "1",
  "--x0", "0.5", "--x1", "1",
  "--numerical-derivative",
  "--export-report", "$output_dir/report.md",
  "--export-json-dir", "$output_dir/traces"
)

python @args