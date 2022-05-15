#!/usr/bin/fish

# Fish shell script use for split dataset into correct directory structure like
# ```
# data/
#     train/
#         dogs/
#             dog001.jpg
#             dog002.jpg
#             ...
#         cats/
#             cat001.jpg
#             cat002.jpg
#             ...
#     validation/
#         dogs/
#             dog001.jpg
#             dog002.jpg
#             ...
#         cats/
#             cat001.jpg
#             cat002.jpg
#             ...
# ```

set dataset_dir "./Flickr8k/Images"
set out_dir "subflickr"
if test -d $out_dir
    rm -rf $out_dir
end
mkdir -p $out_dir/train/imgs
mkdir -p $out_dir/test/imgs
mkdir -p $out_dir/val/imgs

# set tot (count $dataset_dir/*)                                                                                                     
set tot 5500 # limit to 5000
set n_test (math $tot\*.2)
set n_val (math \($tot-$n_test\)\*.2)
set n_train (math $tot-$n_test-$n_val)
set i 0
for file in $dataset_dir/*
   if test $i -eq $tot
       break
   end
   if test $i -lt $n_train
	   cp $file $out_dir/train/imgs
   else if test $i -lt (math $n_train+$n_val)
	   cp $file $out_dir/val/imgs
   else
	   cp $file $out_dir/test/imgs
   end
   set i (math $i+1)
end
