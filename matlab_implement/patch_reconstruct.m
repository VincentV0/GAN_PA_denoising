function img_reconstr = patch_reconstruct(patches, image_size)
    %PATCH_RECONSTRUCT Patches are reconstructed to form a full image
    %   Expected patch-array size: (patches, x, y)
    patches_x = idivide ( int32(image_size(1)) , int32(size(patches, 2)) );
    patches_y = idivide ( int32(image_size(2)) , int32(size(patches, 3)) );

    % Calculate number of images (should be 1!)
    nr_of_images   = idivide ( int32(size(patches,1)) , int32(patches_x * patches_y));
    
    % Define data array
    img_reconstr = zeros(nr_of_images, image_size(1), image_size(2));
    
    i = 1;
    for x=0:patches_x-1
    for y=0:patches_y-1
        img_reconstr(:, size(patches, 2)*x+1:size(patches, 2)*(x+1) , ...
            size(patches, 3)*y+1:size(patches, 3)*(y+1)) = patches(i, :, :);
        i = i+1;
    end
    end

end

