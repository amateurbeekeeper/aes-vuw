#!/bin/bash

echo "================================"
echo "install project (python) dependencies"
echo "================================"

pip3 install -r requirements.txt

echo "================================"
echo "runner.sh started"
echo "================================"

feature_extractor="AES/feature_extractor.py"
pcg_permutate_agreement_test="AES/prediction_config_generator.py"

echo "================================"
echo "1. feature_extractor runs started"
echo "================================"

"python3" $feature_extractor hewlett overall 1 1787 3 > "hewlett_overall_1_1787_3".txt;
"python3" $feature_extractor hewlett overall 3 1726 3 > "hewlett_overall_3_1726_3".txt;
"python3" $feature_extractor hewlett overall 4 1772 3 > "hewlett_overall_4_1772_3".txt;
"python3" $feature_extractor hewlett overall 5 1805 3 > "hewlett_overall_5_1805_3".txt;
"python3" $feature_extractor hewlett overall 6 1800 3 > "hewlett_overall_6_1800_3".txt;
"python3" $feature_extractor hewlett categorized 2 1800 4 > "hewlett_categorized_2_1800_4".txt;
"python3" $feature_extractor hewlett categorized 7 1569 7 > "hewlett_categorized_7_1569_7".txt;
"python3" $feature_extractor hewlett categorized 8 723 9 > "hewlett_categorized_8_723_9".txt;

"python3" $feature_extractor vuw categorized 1 113 8 > "hewlett_overall_1_1787_3".txt;

echo "================================"
echo "================================"
echo "2. pcg_permutate_agreement_test runs started"
echo "================================"
echo "================================"

"python3" $pcg_permutate_agreement_test hewlett overall 1 1787 > "hewlett_overall_1_1787".txt;
"python3" $pcg_permutate_agreement_test hewlett overall 3 1726 > "hewlett_overall_3_1726".txt;
"python3" $pcg_permutate_agreement_test hewlett overall 4 1772 > "hewlett_overall_4_1772".txt;
"python3" $pcg_permutate_agreement_test hewlett overall 5 1805 > "hewlett_overall_5_1805".txt;
"python3" $pcg_permutate_agreement_test hewlett overall 6 1800 > "hewlett_overall_6_1800".txt;
"python3" $pcg_permutate_agreement_test hewlett categorized 2 1800 > "hewlett_categorized_2_1800".txt;
"python3" $pcg_permutate_agreement_test hewlett categorized 7 1569 > "hewlett_categorized_7_1569".txt;
"python3" $pcg_permutate_agreement_test hewlett categorized 8 723 > "hewlett_categorized_8_723".txt;

"python3" $pcg_permutate_agreement_test vuw categorized 1 113 > "vuw_categorized_3_113".txt;






