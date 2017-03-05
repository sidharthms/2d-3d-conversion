--
-- Created by IntelliJ IDEA.
-- User: sidharth
-- Date: 3/5/16
-- Time: 2:48 PM
-- To change this template use File | Settings | File Templates.
--
-- Modified from https://github.com/karpathy/char-rnn

require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'optim'
require 'util'
torch.setnumthreads(6)

debugger = require 'fb.debugger'
eh = require 'fb.util.error'

-- require 'util.misc'
local opts = require 'opts'
local models = require 'models/init'
local Trainer = require 'train'
local GanTrainer = require 'train_gan'
local tds = require 'tds'
-- local checkpoints = require 'checkpoints'

tablex = require('pl.tablex')
stringx = require('pl.stringx')
keys = tablex.keys

local opt = opts.parse(arg)

-- if opt.gpuid >= 0 then
--     require 'cutorch'
--     print('using CUDA on GPU ' .. opt.gpuid .. '...')
--     cutorch.setDevice(opt.gpuid + 1)
-- end

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.manual_seed)
cutorch.manualSeedAll(opt.manual_seed)

-- create the data loader class
local BatchLoader = require 'BatchLoader'
loader = BatchLoader.create(checkpoint, opt)
opt = tablex.update(opt, loader:inferred_opts())

-- graph.dot(model.fg, 'gmodule', 'gmodule')
-- nngraph.simple_print.dot(model.fg, 'gmod', 'gmod')
-- debugger.enter()
-- Model and criterion

-- Load previous checkpoint, if it exists
-- local checkpoint, optimState = checkpoints.latest(opt)

local model, modelD, featureModel, criterion, criterionD
if opt.saved_model ~= '' then
    local model_checkpoint = torch.load(opt.saved_model)
    model = model_checkpoint.model
    featureModel = model_checkpoint.featureModel
    criterion = model_checkpoint.criterion
    optimState = model_checkpoint.optimState
else
    -- Create model
    model, featureModel, criterion = models.setup(opt, checkpoint)
end

if opt.gan_model ~= '' then
    modelD = require('models/' .. opt.gan_model)(opt)
    criterionD = nn.BCECriterion():cuda()
end

-- The trainer handles the training loop and evaluation on validation set
local trainer
if opt.gan_model ~= '' then
    trainer = GanTrainer(model, featureModel, criterion, opt, modelD, criterionD)
else
    trainer = Trainer(model, featureModel, criterion, opt)
end

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

local best1Err = math.huge
for epoch = 1, opt.epochs do
    -- Train for a single epoch
    trainer:run(true, epoch, loader, 1, nil, nil, nil, opt.batch_limit)

    -- Run model on validation set
    local err
    if not opt.no_eval then
        err = trainer:run(false, epoch, loader, 2) --, nil, nil, nil, 100)
    else
        err = 100
    end

    local checkpoint = {}
    checkpoint.model = deepCopy(model):float():clearState()
    checkpoint.featureModel = deepCopy(featureModel):float():clearState()
    checkpoint.criterion = deepCopy(criterion):float()
    if opt.gan_model ~= '' then
        checkpoint.modelD = deepCopy(modelD):float():clearState()
        checkpoint.criterionD = deepCopy(criterionD):float()
    end
    checkpoint.opt = opt

    -- Save the model if it has the best top-1 error
    if err < best1Err and opt.savefile ~= '' and not opt.nosave then
        print(' * Saving best model ', err)
        torch.save(opt.savefile .. '.t7', checkpoint)
        best1Err = err
    elseif err < best1Err then
        print(' * Best model ', err)
        best1Err = err
    end
    if opt.save_every_epoch then
        torch.save(opt.savefile .. '_epoch_' .. epoch .. '.t7', checkpoint)
    end

    -- Run model on test set
    -- local err = trainer:run(false, epoch, loader, 3)
end
