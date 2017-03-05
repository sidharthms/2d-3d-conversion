--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

require 'image'

local M = {}

function M.ComposePair(transforms)
   return function(input1, input2)
      for it, transform in ipairs(transforms) do
         input1, input2 = transform(input1, input2)
      end
      return input1, input2
   end
end

function M.ColorNormalizePair(meanstd)
   return function(img1, img2)
      img1 = img1:clone()
      img2 = img2:clone()
      for i=1,3 do
         img1[i]:add(-meanstd.mean[i])
         img1[i]:div(meanstd.std[i])
         img2[i]:add(-meanstd.mean[i])
         img2[i]:div(meanstd.std[i])
      end
      return img1, img2
   end
end

-- Scales the smaller edge to size
function M.ScalePair(size, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(input1, input2)
      local w, h = input1:size(3), input1:size(2)
      if (w <= h and w == size) or (h <= w and h == size) then
         return input1, input2
      end
      if w < h then
         return image.scale(input1, size, h/w * size, interpolation),
                image.scale(input2, size, h/w * size, interpolation)
      else
         return image.scale(input1, w/h * size, size, interpolation),
                image.scale(input2, w/h * size, size, interpolation)
      end
   end
end

-- Random crop with size 50%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
function M.RandomSizedCropPair(xsize, ysize)
   return function(input1, input2)
      local attempt = 0
      repeat
         local area = input1:size(2) * input1:size(3)
         local targetArea = torch.uniform(0.50, 1.0) * area

         local aspectRatio = torch.uniform(2.2, 2.6)
         local w = torch.round(math.sqrt(targetArea * aspectRatio))
         local h = torch.round(math.sqrt(targetArea / aspectRatio))

         if torch.uniform() < 0.5 then
            w, h = h, w
         end

         if h <= input1:size(2) and w <= input1:size(3) then
            local y1 = torch.random(0, input1:size(2) - h)
            local x1 = torch.random(0, input1:size(3) - w)

            local out1 = image.crop(input1, x1, y1, x1 + w, y1 + h)
            local out2 = image.crop(input2, x1, y1, x1 + w, y1 + h)
            assert(out1:size(2) == h and out1:size(3) == w, 'wrong crop size')

            return image.scale(out1, xsize, ysize, 'bicubic'), image.scale(out2, xsize, ysize, 'bicubic')
         end
         attempt = attempt + 1
      until attempt >= 10

      -- fallback
      print('Fallback')
      return M.RandomCropPair(xsize, ysize)(input1, input2)
   end
end

-- Random crop form larger image with optional zero padding
function M.RandomCropPair(xsize, ysize)
   return function(input1, input2)
      assert(input1:size(3) == input2:size(3) and input1:size(2) == input2:size(2), 'input size mismatch')
      local w, h = input1:size(3), input1:size(2)
      if w == xsize and h == ysize then
         return input1, input2
      end

      local x1, y1 = torch.random(0, w - xsize), torch.random(0, h - ysize)
      local out1 = image.crop(input1, x1, y1, x1 + xsize, y1 + ysize)
      local out2 = image.crop(input2, x1, y1, x1 + xsize, y1 + ysize)
      assert(out1:size(3) == xsize and out1:size(2) == ysize, 'wrong crop size')
      assert(out2:size(3) == xsize and out2:size(2) == ysize, 'wrong crop size')
      return out1, out2
   end
end

-- Lighting noise (AlexNet-style PCA-based noise)
function M.LightingPair(alphastd, eigval, eigvec)
   return function(input1, input2)
      if alphastd == 0 then
         return input1, input2
      end

      local alpha = torch.Tensor(3):normal(0, alphastd)
      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()

      input1 = input1:clone()
      input2 = input2:clone()
      for i=1,3 do
         input1[i]:add(rgb[i])
         input2[i]:add(rgb[i])
      end
      return input1, input2
   end
end

local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function M.SaturationPair(var)
   local gs1
   local gs2

   return function(input1, input2)
      gs1 = gs1 or input1.new()
      gs2 = gs2 or input2.new()
      grayscale(gs1, input1)
      grayscale(gs2, input2)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input1, gs1, alpha)
      blend(input2, gs2, alpha)
      return input1, input2
   end
end

function M.BrightnessPair(var)
   local gs1
   local gs2

   return function(input1, input2)
      gs1 = gs1 or input1.new()
      gs2 = gs2 or input2.new()
      gs1:resizeAs(input1):zero()
      gs2:resizeAs(input2):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input1, gs1, alpha)
      blend(input2, gs2, alpha)
      return input1, input2
   end
end

function M.ContrastPair(var)
   local gs1
   local gs2

   return function(input1, input2)
      gs1 = gs1 or input1.new()
      gs2 = gs2 or input2.new()
      grayscale(gs1, input1)
      grayscale(gs2, input2)
      gs1:fill(gs1[1]:mean())
      gs2:fill(gs2[1]:mean())

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input1, gs1, alpha)
      blend(input2, gs2, alpha)
      return input1, input2
   end
end

function M.RandomOrderPair(ts)
   return function(input1, input2)
      local img1 = input1.img or input1
      local img2 = input2.img or input2
      local order = torch.randperm(#ts)
      for i=1,#ts do
         img1, img2 = ts[order[i]](img1, img2)
      end
      return input1, input2
   end
end

function M.ColorJitterPair(opt)
   local brightness = opt.brightness or 0
   local contrast = opt.contrast or 0
   local saturation = opt.saturation or 0

   local ts = {}
   if brightness ~= 0 then
      table.insert(ts, M.BrightnessPair(brightness))
   end
   if contrast ~= 0 then
      table.insert(ts, M.ContrastPair(contrast))
   end
   if saturation ~= 0 then
      table.insert(ts, M.SaturationPair(saturation))
   end

   if #ts == 0 then
      return function(input1, input2) return input1, input2 end
   end

   return M.RandomOrderPair(ts)
end

return M
