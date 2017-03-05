local nn = require 'nn'
local debugger = require 'fb.debugger'

require 'cunn'
require 'cudnn'
require 'nngraph'

local function createModel(opt)
	local input_nc, output_nc, ndf, n_layers
	local input_nc = 3
	local output_nc = 3
	local n_layers = 2
	local ndf = opt.gan_d_hidden_size
	if opt.no_cgan then
		input_nc = 0
	end

	local netD = nn.Sequential()

	-- input is (nc) x 160 x 184
	netD:add(cudnn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1))
	netD:add(nn.LeakyReLU(0.2, true))

	local nf_mult = 1
	local nf_mult_prev
	for n = 1, n_layers do
	   nf_mult_prev = nf_mult
	   nf_mult = math.min(2^n,8)
	   netD:add(cudnn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 2, 2, 1, 1))
	   netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
	end

	-- state size: (ndf*M) x N x N
	-- state size for 3 layers: 256 x 20 x 48
	if opt.gan_d_extra_layer then
		nf_mult_prev = nf_mult
		nf_mult = math.min(2^(n_layers+1),8)
		netD:add(cudnn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 3, 3, 1, 1, 1, 1))
		netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
	end
	-- -- state size: 256 x 20 x 48
	netD:add(cudnn.SpatialConvolution(ndf * nf_mult, 1, 3, 3, 1, 1, 1, 1))
	-- state size: 1 x 20 x 48

	netD:add(cudnn.Sigmoid())
	-- state size: 1 x 20 x 48

	return nn.GPU(netD, 2):cuda()
end

return createModel
