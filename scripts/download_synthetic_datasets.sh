mkdir -p data/datasets
cd data/datasets

# Download Synthetic dataset
for dataset in "PS_Sculpture_Dataset.tgz" "PS_Blobby_Dataset.tgz"; do
    echo "Downloading $dataset"
    wget https://huggingface.co/Guanying/PhotometricStereo_Blobby_Sculpture_Dataset/resolve/main/PS_Sculpture_Dataset.tgz.partaa
    wget https://huggingface.co/Guanying/PhotometricStereo_Blobby_Sculpture_Dataset/resolve/main/PS_Sculpture_Dataset.tgz.partab
    wget https://huggingface.co/Guanying/PhotometricStereo_Blobby_Sculpture_Dataset/resolve/main/PS_Blobby_Dataset.tgz
    tar -xvf $dataset
    rm $dataset
done

# Back to root directory
cd ../../

