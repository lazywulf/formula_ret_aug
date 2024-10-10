param (
    [string]$dataset,
    [string]$result
)

function check_status {
    param (
        [string]$command_name
    )
    if ($LASTEXITCODE -eq 0) {
        Write-Host "oooooooooooooooo    $command_name finished   oooooooooooooooo"
    } else {
        Write-Host "xxxxxxxxxxxxxxxx      $command_name error    xxxxxxxxxxxxxxxx"
    }
}

function check_dataset {
    param (
        [string]$root
    )
    $required_files = @(
        "opt_char_embedding.txt",
        "opt_list.txt",
        "opt_judge.txt",
        "query_opt_list.txt",
        "slt_char_embedding.txt",
        "slt_list.txt",
        "slt_judge.txt",
        "query_slt_list.txt"
    )
    try {
        foreach ($file in $required_files) {
            $file_path = Join-Path -Path $root -ChildPath $file
            if (-Not (Test-Path -Path $file_path)) {
                Write-Host "Missing required file: $file"
                return
            }
        }
        Write-Host "All required files are present"
    }
    catch {
        Write-Host "An error occurred while checking the dataset: $_"
        return
    }
}

function get_swap_dict {
    param (
        [string]$dataset
    )
    $command = "python get_swap_dict.py $dataset"
    Invoke-Expression $command
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Swap dict created"
    } else {
        Write-Host "An error occurred while creating the swap dict"
    }
}

function train_GCL {
    param (
        [string]$bs,
        [string]$run_id,
        [string]$aug_type,
        [string]$root,
        [string]$result_root
    )
    $command1 = "python train_query_GCL_slt_or_opt.py --encode opt --pretrained --run_id $run_id --aug_id $aug_type --bs $bs --dataset $root --result $result_root"
    $command2 = "python train_query_GCL_slt_or_opt.py --encode slt --pretrained --run_id $run_id --aug_id $aug_type --bs $bs --dataset $root --result $result_root"

    Invoke-Expression $command1
    check_status $command1
    Invoke-Expression $command2
    check_status $command2
}


$run_ids = 1..5
$batch_sizes = 256, 512, 1024, 2048, 4096, 8192
$aug_types = 1..6

check_dataset -root $dataset
get_swap_dict -dataset $dataset

foreach ($aug_type in $aug_types) {
    foreach ($bs in $batch_sizes) {
        foreach ($run_id in $run_ids) {
            train_GCL -run_id $run_id -aug_type $aug_type -bs $bs -root $dataset -result $result
        }
    }
}
