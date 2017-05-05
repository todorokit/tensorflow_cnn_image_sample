<?php

$sep = DIRECTORY_SEPARATOR;

$sourceDir='img';
$trainDir='train';
$testDir='test';

$trFile='train.txt';
$teFile='test.txt';
$mapFile='config2.py';

@unlink($trFile);
if(!@touch($trFile)) {
    echo $tfFile,"\n";
    exit;
}

@unlink($teFile);
if(!@touch($teFile)) {
    echo $teFile,"\n";
    exit;
}

function myscandir($dir) {
    $files = [];
    exec("c:\\cygwin64\\bin\\ls.exe {$dir}", $files);
    $ret = [];
    foreach($files as $file) {
        if($file == "." || $file == "..") {
            continue;
        }
        $ret[] = $file;
    }
    return $ret;
}

$dirs = myscandir($sourceDir); 

function getId ($key) {
    global $dirs;
    return array_search($key, $dirs);
}

# lsとsortを組み合わせてランダムに処理
$images = [];
exec("where /R {$sourceDir} *", $images);
shuffle($images);

file_put_contents($mapFile, "classList = {}\n");
foreach($dirs as $key => $wordId) {
    file_put_contents($mapFile, "classList[{$key}] = \"{$wordId}\"\n", FILE_APPEND);
}

$limits = [];
$limits2 = [];
$datanum = $argv[1];
$testnum = $argv[2];

foreach($images as $image) {
    $info = pathinfo($image);
    $wordId = basename($info["dirname"]);
    $basename = $info["basename"];
    if(isset($limits2[$wordId]) && $limits2[$wordId] >= $testnum) {
        continue;
    } else if(isset($limits[$wordId]) && $limits[$wordId] >= $datanum) {
        if(!isset($limits2[$wordId])) {
            $limits2[$wordId] = 0;
        }
        $limits2[$wordId]++;
        $isTest = 1;
    } else {
        if(!isset($limits[$wordId])) {
            $limits[$wordId] = 0;
        }
        $isTest = 0;
        $limits[$wordId]++;
    }
    $id = getId($wordId);

    if ($isTest) {
        file_put_contents($teFile, "{$image} {$id}\n", FILE_APPEND);
    } else {
        file_put_contents($trFile, "{$image} {$id}\n", FILE_APPEND);
    }
}
