function data_arr = patch_extract(img,patch_size)
    %PATCH_EXTRACT Extract patches from an image
    %   Expected image size: (frame, x, y)
    %   Image size / patch size has to be an integer; so, no overlapping!
    
    % assume an image shape of (frame, x, y)
    patches_x = idivide ( int32(size(img,2)) , int32(patch_size(1)) );
    patches_y = idivide ( int32(size(img,3)) , int32(patch_size(2)) );
    
    % calculate number of patches per image
    patches_per_img = patches_x*patches_y;
    assert(mod(size(img,2), patch_size(1)) == 0 | mod(size(img,3), patch_size(2)) == 0 , "data shape divided by patch size should yield an integer");
    
    % Put patches into array
    data_arr = zeros(size(img,1)*patches_per_img, patch_size(1), patch_size(2));
    i = 1;
    for x=0:patches_x-1
    for y=0:patches_y-1
        patch = img(:, patch_size(1)*x+1:patch_size(1)*(x+1) , patch_size(2)*y+1:patch_size(2)*(y+1));
        data_arr(i, :, :) = patch(1,:,:);
        i = i+1;
    end
    end

end

