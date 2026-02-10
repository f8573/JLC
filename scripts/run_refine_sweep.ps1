$mrs = 4..6
$nrs = @(4,8)
$ks = @(4,6,8,10)
foreach ($mr in $mrs) {
  foreach ($nr in $nrs) {
    foreach ($k in $ks) {
      Write-Host "Running MR=$mr NR=$nr K=$k"
      $cmd = ".\\gradlew.bat -Dla.gemm.mr=$mr -Dla.gemm.nr=$nr -Dla.gemm.kunroll=$k -Djava.util.concurrent.ForkJoinPool.common.parallelism=8 runGemmTest"
      Start-Process -NoNewWindow -Wait -FilePath cmd.exe -ArgumentList "/c", $cmd
      Start-Sleep -Seconds 2
    }
  }
}
Write-Host "Refine sweep finished."