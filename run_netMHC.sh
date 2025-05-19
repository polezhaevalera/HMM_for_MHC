#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p netmhcpan_outputs

# Process each .pep file
for input_file in test_peptides_*.pep; do
    # Skip if the file is a .pep.myout file or doesn't exist
    [[ "$input_file" == *.pep.myout ]] && continue
    [ ! -f "$input_file" ] && continue

    # Extract allele information from filename
    # Format: test_peptides_DRB1_11_01_2.pep → DRB1_1101
    IFS='_' read -ra parts <<< "$input_file"
    
    allele_prefix="${parts[2]}"      # e.g., DRB1
    allele_num1="${parts[3]}"        # e.g., 11
    allele_num2="${parts[4]}"        # e.g., 01
    
    # Combine into desired format (DRB1_1101)
    allele_code="${allele_prefix}_${allele_num1}${allele_num2}"
    echo "$allele_code"

    # Create output filename (using .myout extension as in your working example)
    base_name="${input_file%.pep}"
    output_file="netmhcpan_outputs/${base_name}.xls"

    # Run netMHCIIpan using your verified command format
    echo "Processing $input_file with allele $allele_code..."
    ../netMHCIIpan -f "$input_file" -a "$allele_code" -inptype 1 -xls -xlsfile "$output_file"

    echo "Output saved to $output_file"
    echo "----------------------------------"
done

echo "All files processed!"