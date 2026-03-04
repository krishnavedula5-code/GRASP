$output_dir = "outputs/cubic_run"
Remove-Item -Recurse -Force $output_dir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $output_dir | Out-Null

$args = @(
  "-m", "numerical_lab.cli",
  "--expr", "x**3 - x - 2",
  "--dexpr", "3*x**2 - 1",
  "--mode", "compare",
  "--a", "1", "--b", "2",
  "--x0", "1", "--x1", "2",
  "--export-report", "$output_dir/report.md",
  "--export-json-dir", "$output_dir/traces"
)

python @args