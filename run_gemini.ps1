param(
    [ValidateSet('check', 'preview', 'pilot', 'full')]
    [string]$Mode = 'check'
)
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'memory_misevolution/policy_filter_experiment.py'
$pythonArgs = @($runner, '--provider', 'gemini')
switch ($Mode) {
    'check' { $pythonArgs += '--check-connection' }
    'preview' { $pythonArgs += @('--limit', '2', '--repetitions', '1') }
    'pilot' { $pythonArgs += @('--live', '--limit', '2', '--repetitions', '1') }
    'full' { $pythonArgs += @('--live', '--limit', '10', '--repetitions', '3') }
}
& python @pythonArgs
exit $LASTEXITCODE
