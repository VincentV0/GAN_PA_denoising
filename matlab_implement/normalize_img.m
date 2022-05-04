function img_norm = normalize_img(img)
%NORMALIZE_IMG Normalize an image on a frame-to-frame basis
%   Expected image shape: (frame, x, y)
xmax = max(img, [], [2 3]);
xmin = min(img, [], [2 3]);
img_norm = (img-xmin) ./ (xmax-xmin);
end

