--
-- Created by IntelliJ IDEA.
-- User: sidharth
-- Date: 3/10/16
-- Time: 11:02 PM
-- To change this template use File | Settings | File Templates.
--

local types = require 'pl.types'
local dir = require 'pl.dir'
local path = require 'pl.path'
local Threads = require 'threads'
local t = require 'transforms'

require 'image'
require 'torchzlib'

Threads.serialization('threads.sharedserialize')

local BatchLoader = {}
BatchLoader.__index = BatchLoader

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function save_img(filename, img)
	img = img:clone()
	for i=1,3 do
		img[i]:mul(meanstd.std[i])
		img[i]:add(meanstd.mean[i])
	end
	image.save(filename, img)
end

function file_exists(name)
	local f=io.open(name,"r")
	if f~=nil then io.close(f) return true else return false end
end

function BatchLoader.create_loader(checkpoint, opt)
	local self = {}
	setmetatable(self, BatchLoader)
	self.opt = opt
	self.batch_size = self.opt.batch_size
	self.batch_idx = {0,0,0}
	self.batch_randperm = {nil, nil, nil}
	self.total_load_time = {0,0,0}
	self.files_per_batch = self.batch_size / self.opt.data_split_size
	self.proddata_dir = path.join(self.opt.data_dir, self.opt.proddata_dir)

	assert(self.batch_size % self.opt.data_split_size == 0,
		'batch_size must be a multiple of data_split_size')

	local data_dir = self.opt.data_dir

	-- construct a tensor with all the data
	if not path.exists(self.proddata_dir) then
		dir.makepath(self.proddata_dir)
		self:_preprocess_tensors()
	end

	self.metadata = torch.load(path.join(self.proddata_dir, 'metadata.t7'))

	collectgarbage()
	return self
end

function BatchLoader.create(checkpoint, opt)
	if not opt.parallel_preprocessing then
		return BatchLoader.create_loader(checkpoint, opt)
	end

	local manualSeed = opt.manual_seed
	local loader = BatchLoader.create_loader(checkpoint, opt)

	local function init()
		require 'nn'
		require 'cunn'
		require 'cudnn'
		require 'nngraph'
		require 'pl'
		torch.setnumthreads(1)

		t = require 'transforms'
		BatchLoaderInThread = require 'BatchLoader'
	end

	local function main(idx)
		if manualSeed ~= 0 then
			torch.manualSeed(manualSeed + idx)
		end
		_G.loader = loader
	end

	local threads = Threads(opt.n_threads, init, main)
	loader.threads = threads

	return loader
end

function BatchLoader:inferred_opts()
	return {}
end

function BatchLoader:reset_batch_pointer(split_idx, batch_idx)
	batch_idx = batch_idx or 0
	self.batch_idx[split_idx] = batch_idx
	self.total_load_time[split_idx] = 0
end

function BatchLoader:preprocess(split)
	-- Computed from random subset of ImageNet training images
	local meanstd = {
	   mean = { 0.485, 0.456, 0.406 },
	   std = { 0.229, 0.224, 0.225 },
	}
	local pca = {
	   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
	   eigvec = torch.Tensor{
	      { -0.5675,  0.7192,  0.4009 },
	      { -0.5808, -0.0045, -0.8140 },
	      { -0.5836, -0.6948,  0.4203 },
	   },
	}

   if split == 1 then
      return t.ComposePair{
         t.RandomSizedCropPair(self.opt.xsize, self.opt.ysize),
         t.ColorJitterPair({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.LightingPair(0.1, pca.eigval, pca.eigvec),
         t.ColorNormalizePair(meanstd),
      }
   else
      return t.ComposePair{
         t.ScalePair(self.opt.ysize),
         t.ColorNormalizePair(meanstd),
      }
   end
end

function BatchLoader:load_tensor(path)
	if not self.opt.no_compress then
		return torch.load(path):decompress()
	end
	return torch.load(path)
end

function BatchLoader:next_batch(split_idx, idx)
	if self.load_timer == nil then
		self.load_timer = torch.Timer()
		self.total_load_time[split_idx] = 0
	end
	self.load_timer:reset()

	local x_tensors = {}
	local y_tensors = {}
	for is=1, self.files_per_batch do
		local x_path = path.join(self.proddata_dir, 'x_slice_' .. tostring(split_idx) ..
				'_' .. tostring((idx-1) * self.files_per_batch + is) .. '.t7')
		local y_path = path.join(self.proddata_dir, 'y_slice_' .. tostring(split_idx) ..
				'_' .. tostring((idx-1) * self.files_per_batch + is) .. '.t7')

		local x_tensor = BatchLoaderInThread.load_tensor(self, x_path):float():transpose(3, 4) / 255
		local y_tensor = BatchLoaderInThread.load_tensor(self, y_path):float():transpose(3, 4) / 255
		-- save_img('x.png', x_tensor[1])
		-- save_img('y.png', y_tensor[1])
		-- debugger.enter()

		local x_preprocessed = torch.Tensor(x_tensor:size(1), x_tensor:size(2), self.opt.ysize, self.opt.xsize)
		local y_preprocessed = torch.Tensor(x_tensor:size(1), x_tensor:size(2), self.opt.ysize, self.opt.xsize)
		for i=1, x_tensor:size(1) do
			local preprocessor = BatchLoaderInThread.preprocess(self, split_idx)
			local x, y = preprocessor(x_tensor[i], y_tensor[i])
			x_preprocessed[i]:copy(x)
			y_preprocessed[i]:copy(y)
		end

		table.insert(x_tensors, x_preprocessed)
		table.insert(y_tensors, y_preprocessed)
	end

	local x = torch.cat(x_tensors, 1)
	local y = torch.cat(y_tensors, 1)

	self.total_load_time[split_idx] = self.total_load_time[split_idx] + self.load_timer:time().real

	return x, y, nil
end

function BatchLoader:run(split_idx, randperm)
	if self.opt.parallel_preprocessing then
		return self:run_parallel(split_idx, randperm)
	else
		BatchLoaderInThread = self
		return self:run_sequential(split_idx, randperm)
	end
end

function BatchLoader:run_parallel(split_idx, randperm)
	local threads = self.threads
	local n = 0
	local x, y, d
	self:reset_batch_pointer(split_idx)
	if randperm then
		print('Using random batch permutations')
		self.batch_randperm[split_idx] = torch.randperm(self:batches(split_idx))
	end

	local function enqueue()
		while threads:acceptsjob() do
			self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
			if self.batch_idx[split_idx] > self:batches(split_idx) then
				return
			end
			local batch_idx = self.batch_idx[split_idx]
			if randperm then
				batch_idx = self.batch_randperm[split_idx][batch_idx]
			end
			threads:addjob(
				function(split_idx, batch_idx)
					return BatchLoaderInThread.next_batch(_G.loader, split_idx, batch_idx)
				end,
				function(_x, _y, _d)
					x = _x
					y = _y
					d = _d
				end,
				split_idx,
				batch_idx
			)
		end
	end

	return function()
		enqueue()
		if not threads:hasjob() then
			return nil
		end
		threads:dojob()
		if threads:haserror() then
			threads:synchronize()
		end
		enqueue()
		n = n + 1
		return n, x, y, d
	end
end

function BatchLoader:run_sequential(split_idx, randperm)
	local n = 0
	self:reset_batch_pointer(split_idx)
	self.preprocessor = self:preprocess(split_idx)
	if randperm and split_idx == 1 then
		print('Using random batch permutations')
		self.batch_randperm[split_idx] = torch.randperm(self.num_batches[split_idx])
	end
	return function()
		n = n + 1

		-- split_idx is integer: 1 = train, 2 = val, 3 = test
		self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
		if self.batch_idx[split_idx] > self:batches(split_idx) then
			print(('Load time in Epoch: %.3f'):format(self.total_load_time[split_idx]))
			self.total_load_time[split_idx] = 0
			return nil
		end
		local idx = self.batch_idx[split_idx]
		if randperm and split_idx == 1 then
			idx = self.batch_randperm[split_idx][idx]
		end

		local x, y, d = self:next_batch(split_idx, idx)
		if x == nil then
			print('NEXT BATCH RETURNED NIL')
			return nil
		else
			return n, x, y, d
		end
	end
end

function BatchLoader:batches(split_idx)
	return self.metadata.file_count[split_idx]-1

	-- if self.opt.num_batches ~= nil then
	-- 	self.num_batches = self.opt.num_batches
	-- end
	--
	-- self.num_batches = self.num_batches or {}
	-- if self.num_batches[split_idx] == nil then
	-- 	local proddata_files = dir.getfiles(self.proddata_dir, 'x_slice_' .. tostring(split_idx) .. '*')
	-- 	self.num_batches[split_idx] = #proddata_files
	-- end
	-- return self.num_batches[split_idx]
end

-- Save groups of self.opt.data_split_size frames to disk, ignore last few frames.
function BatchLoader:_save_splits(remainder, frames, split_idx, prefix, mat_index)
	if remainder ~= nil then
		frames = torch.cat(remainder, frames, 1)
	end
	local num_frames = self.opt.data_split_size * math.floor(frames:size(1) / self.opt.data_split_size)
	local save_subset = frames:sub(1, num_frames)
	local remainder_frames = frames:sub(num_frames + 1, frames:size(1))

	local file_num = self._file_num[split_idx]
	if split_idx == 1 then
		if self._perm[split_idx][mat_index] == nil then
			print('Shuffling...')
			local perm = torch.randperm(num_frames):long()
			self._perm[split_idx][mat_index] = perm
			save_subset = save_subset:index(1, perm)
			print('Shuffled.')
		else
			print('Using previously gen perm')
			local perm = self._perm[split_idx][mat_index]
			save_subset = save_subset:index(1, perm)
		end
	end

	local splits = math.floor(save_subset:size(1) / self.opt.data_split_size)
	print('Converting mat file for split', tostring(split_idx), '...')
	for ib=1, splits do
		io.write(('\r[%3.2f][%10d/%d]'):format(ib / splits, ib, splits))
		local start_idx = (ib-1) * self.opt.data_split_size + 1
		local end_idx = ib * self.opt.data_split_size
		local slice = save_subset:sub(start_idx, end_idx):clone()
		if not self.opt.no_compress then
			slice = torch.CompressedTensor(slice)
		end
		torch.save(path.join(self.proddata_dir, prefix .. '_slice_' .. tostring(split_idx) ..
				'_' .. tostring(file_num) .. '.t7'), slice)
		file_num = file_num + 1
	end
	self._file_num[split_idx] = file_num
	print('')

	return remainder_frames
end

function BatchLoader:_preprocess_tensors(out_tensorfile)
	print('Processing tensors...')
	require 'mattorch'

	local filenames = dir.getfiles(self.opt.data_dir, 'train_left*')
	local file_order = torch.randperm(#filenames)
	local size_hash = {}

	self._perm = {{}, {}}
	self._file_num = {1, 1}
	local remainder_frames = nil
	print('Test files:', file_order:sub(self.opt.train_mats+1, #filenames))
	for ix=1, #filenames do
		local xfile = path.join(self.opt.data_dir, 'train_left' .. tostring(file_order[ix]) .. '.mat')
		if not path.exists(xfile) then
			size_hash[ix] = 0
			print(xfile, 'Does not exist')
		else
			print('Loading x input', xfile)
			local loaded = tablex.values(mattorch.load(xfile))[1]
			local split_idx = ix <= self.opt.train_mats and 1 or 2
			remainder_frames = self:_save_splits(remainder_frames, loaded, split_idx, 'x', ix)
			size_hash[ix] = self._file_num[1] + self._file_num[2]
		end
		-- if ix == 1 then
		-- 	break
		-- end
	end
	collectgarbage()

	self._file_num = {1, 1}
	remainder_frames = nil
	for iy=1, #filenames do
		local yfile = path.join(self.opt.data_dir, 'train_right' .. tostring(file_order[iy]) .. '.mat')
		if not path.exists(yfile) then
			assert(size_hash[iy] == 0)
			print(yfile, 'Does not exist')
		else
			print('Loading y input', yfile)
			local loaded = tablex.values(mattorch.load(yfile))[1]
			local split_idx = iy <= self.opt.train_mats and 1 or 2
			remainder_frames = self:_save_splits(remainder_frames, loaded, split_idx, 'y', iy)
			if size_hash[iy] ~= self._file_num[1] + self._file_num[2] then
				debugger.enter()
			end
		end
		-- if iy == 1 then
		-- 	break
		-- end
	end
	collectgarbage()

	local metadata = {file_count=self._file_num}
	torch.save(path.join(self.proddata_dir, 'metadata.t7'), metadata)

	debugger.enter()

	print "Initial loading done"
end

return BatchLoader
