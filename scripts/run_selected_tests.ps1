$j = Get-ChildItem -Path $env:USERPROFILE\.gradle -Recurse -Filter 'junit-4.13.2.jar' -ErrorAction SilentlyContinue | Select-Object -First 1
$h = Get-ChildItem -Path $env:USERPROFILE\.gradle -Recurse -Filter 'hamcrest-core-1.3.jar' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $j) { Write-Host 'junit jar not found'; exit 2 }
if (-not $h) { Write-Host 'hamcrest jar not found'; exit 3 }

# Default tests if none provided as arguments
if ($args.Length -gt 0) {
    $tests = $args
} else {
    $tests = @('net.faulj.stress.HouseholderQRTest','net.faulj.stress.HessenbergDebugTest')
}

$cp = "build\classes\java\test;build\classes\java\main;$($j.FullName);$($h.FullName)"
Write-Host "Classpath:`n$cp"

# Build Java arguments and append test class names
$javaArgs = @("--add-modules=jdk.incubator.vector","--enable-preview","-cp",$cp,"org.junit.runner.JUnitCore") + $tests

& java $javaArgs
