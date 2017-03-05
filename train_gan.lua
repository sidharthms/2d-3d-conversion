--
--  This code is based on https://github.com/facebook/fb.resnet.torch
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'
local tablex = require('pl.tablex')
local listx = require('pl.List')
local dir = require 'pl.dir'
local path = require 'pl.path'

local M = {}
local Trainer = torch.class('resnet.GanTrainer', M)

local REAL_LABEL = 1
local FAKE_LABEL = 0
local PRINT_LOSS_FACTOR = 1

local function clip_gradients(grads, max_grad_norm, max_grad)
    local grad_norm = torch.norm(grads)
    local scale_num = max_grad_norm
    local scale_den = math.max(max_grad_norm, grad_norm)
    grads = grads * (scale_num / scale_den)
    if grads:max() > 10 then
        print('factor = ', (scale_num / scale_den))
    end
    grads = torch.clamp(grads, -max_grad, max_grad)
    return grads
end

function Trainer:__init(modelG, featureModel, criterionG, opt, modelD, criterionD)
    self.modelG = modelG
    self.modelD = modelD
    self.featureModel = featureModel
    self.criterionG = criterionG
    self.criterionD = criterionD
    self.opt = opt

    if opt.optim == 'adam' then
        self.optimStateG = {
            learningRate = opt.lr,
            beta1 = opt.beta1,
            weightDecay = opt.weight_decay
        }
        self.optimStateD = {
            learningRate = opt.d_lr_factor * opt.lr,
            beta1 = opt.beta1,
            weightDecay = opt.weight_decay
        }
    else
        error('Unknown optim type')
    end

    self.paramsG, self.rawGradParamsG = modelG:getParameters()
    self.paramsD, self.rawGradParamsD = modelD:getParameters()
    print('Size gradParamsG:', self.rawGradParamsG:size())
    print('Size gradParamsD:', self.rawGradParamsD:size())

    if self.opt.batch_size_factor ~= 1 then
        self.gradParamsG = self.rawGradParamsG:clone():zero()
        self.gradParamsD = self.rawGradParamsD:clone():zero()
    else
        self.gradParamsG = self.rawGradParamsG
        self.gradParamsD = self.rawGradParamsD
    end

    -- Clear old output images
    local sample_dir = path.join(self.opt.data_dir, self.opt.samples_dir)
    if self.opt.samples_dir ~= '' and path.exists(sample_dir) then
        dir.rmtree(sample_dir)
    end

    self.fakePair = torch.CudaTensor()
    self.realPair = torch.CudaTensor()
end

function Trainer:getFeval(name)
    return function()
        local f = self['criterionOut' .. name]
        local params = self['params' .. name]
        local gradParams = self['gradParams' .. name]

        -- clip gradients
        local clipped_gradients = gradParams

        if self.opt.clip_grads then
            clipped_gradients = clip_gradients(gradParams, self.opt.grad_max_norm, self.opt.grad_max)
        end

        local stats_str = (name .. ': %.3f, max: %.3f, min: %.3f, mean: %.3f; gmax: %.3f, gmin: %.3f'
                .. ', gmean: %.3f'):format(f, params:max(), params:min(), params:mean(),
                clipped_gradients:max(), clipped_gradients:min(), clipped_gradients:mean())

        if not self.opt.no_print_stats then
            print(stats_str)
        end

        return f, clipped_gradients
    end
end

function Trainer:createPairs(gOut)
    if not self.opt.no_cgan then
        torch.cat(self.fakePair, self.input, gOut, 2)
        torch.cat(self.realPair, self.input, self.target, 2)
    else
        self.fakePair = self.target
        self.realPair = gOut
    end
end

function Trainer:dPass(pair, label)
    local dOut = self.modelD:forward(pair)
    local labelTensor = torch.CudaTensor(dOut:size()):fill(label)
    local errDreal = self.criterionD:forward(dOut, labelTensor)
    local df = self.criterionD:backward(dOut, labelTensor)
    return errDreal, df
end

function Trainer:should_skip(batch)
    return false
end

function Trainer:run(train, epoch, dataloader, split_idx, threshold, mistakes_filename, batch_callback, batch_limit,
        predictions_filename)
    assert(train ~= nil, 'train can\'t be nil')
    assert(split_idx ~= nil, 'Split index can\'t be nil')
    if train then
        assert(split_idx == 1, 'Cannot train over a data set other than training set')

        -- Trains the model for a single epoch
        self.optimStateG.learningRate = self:learningRate(epoch)
        self.optimStateD.learningRate = self.opt.d_lr_factor * self:learningRate(epoch)
        print('Learing rate for epoch ' .. epoch .. ': ' .. self.optimStateG.learningRate)
    end

    local set_type
    if split_idx == 1 then
        set_type = 'Train'
    elseif split_idx == 2 then
        set_type = 'Validation'
    else
        set_type = 'Test'
    end

    local run_type
    if train then
        run_type = 'TRAINING'
    else
        run_type = 'EVALUATING'
    end

    local timer = torch.Timer()
    local dataTimer = torch.Timer()

    local totalDataTime = 0
    local lossL1Sum = 0.0
    local lossGSum = 0.0
    local lossDSum = 0.0
    local tpSum = 0.0
    local fpSum = 0.0
    local fnSum = 0.0
    local N = 0
    local batches = dataloader:batches(split_idx)
    local batch_limit = batch_limit and batch_limit > 0 and batch_limit or batches
    local all_outputs = {}

    print('=> ' .. run_type .. ' Epoch # ' .. epoch)
    -- set the batch norm to training or evaluation mode
    if train then
        self.modelG:training()
        self.modelD:training()
    else
        self.modelG:evaluate()
        self.modelD:evaluate()
    end

    local effBatch = 1
    for batch, x, y, d in dataloader:run(split_idx, self.opt.batch_randperm) do
        -- if batch == 2 then
        --     break
        -- end
        if not self:should_skip(batch) then
            totalDataTime = totalDataTime + dataTimer:time().real

            -- x = x:sub(1, 48)
            -- y = y:sub(1, 48)

            -- self.model:clearState()
            self:copyInputs(x, y, d)

            -- if batch == 1 then
            --     local features = eh.on_error(self.featureModel.forward, standard_eh, self.featureModel, self.input)
            --     local raw_out = eh.on_error(self.model.forward, standard_eh, self.model, {self.input, features})
            -- end

            local features = self.featureModel:forward(self.input)
            local gOut = self.modelG:forward({self.input, features}):cuda()
            self:createPairs(gOut)

            -- print('Verify out')
            if self.opt.samples_dir ~= '' and batch % self.opt.sample_save_rate == 0 then
                dir.makepath(path.join(self.opt.data_dir, self.opt.samples_dir))
                local suffix = '_' .. set_type .. '_' .. tostring(epoch) .. '_' .. tostring(batch)
                save_img(path.join(self.opt.data_dir, self.opt.samples_dir, 'input' .. suffix .. '.ppm'), self.input[1])
                save_img(path.join(self.opt.data_dir, self.opt.samples_dir, 'prediction' .. suffix .. '.ppm'), gOut[1])
                save_img(path.join(self.opt.data_dir, self.opt.samples_dir, 'target' .. suffix .. '.ppm'), self.target[1])
            end
            local lossL1 = PRINT_LOSS_FACTOR * 100 * self.criterionG:forward(gOut, self.target)
            lossL1Sum = lossL1Sum + lossL1

            if self.paramsG:ne(self.paramsG):sum() > 0 or self.paramsD:ne(self.paramsD):sum() > 0 then
                print('nan detected')
                debugger.enter()
            end

            -- local err = self:computeScore(raw_out, y, threshold)

            local doGradUpdate = false
            if train then
                if effBatch % self.opt.d_update_freq == 0 then
                    self.modelD:zeroGradParameters()

                    local errDreal, df = self:dPass(self.realPair, REAL_LABEL)
                    self.modelD:backward(self.realPair, df)

                    local errDfake, df = self:dPass(self.fakePair, FAKE_LABEL)
                    self.modelD:backward(self.fakePair, df)

                    self.criterionOutD = errDreal + errDfake
                end

                self.modelG:zeroGradParameters()
                local errD, dfDo = self:dPass(self.fakePair, REAL_LABEL)
                local dfDg = self.modelD:updateGradInput(self.fakePair, dfDo):narrow(2, self.fakePair:size(2)-2, 3)
                local dfDoL1 = self.criterionG:backward(gOut, self.target)
                self.modelG:backward({self.input, features}, dfDg + dfDoL1:mul(self.opt.gan_l1_factor))
                self.criterionOutG = errD + self.opt.gan_l1_factor * self.criterionG.output
                -- debugger.enter()

                if self.opt.batch_size_factor ~= 1 then
                    batch_phase = N % self.opt.batch_size_factor + 1
                    if batch_phase == 1 then
                        self.gradParamsG:zero()
                        self.gradParamsD:zero()
                    end
                    -- debugger.enter()
                    self.gradParamsG:add(self.rawGradParamsG)
                    self.gradParamsD:add(self.rawGradParamsD)

                    if batch_phase == self.opt.batch_size_factor or batch == batch_limit then
                        -- debugger.enter()
                        self.gradParamsG:div(batch_phase)
                        self.gradParamsD:div(batch_phase)
                        doGradUpdate = true
                    end
                else
                    doGradUpdate = true
                end

                if doGradUpdate then
                    if self.gradParamsG:ne(self.gradParamsG):sum() > 0 or self.gradParamsD:ne(self.gradParamsD):sum() > 0 then
                        print('nan detected')
                        debugger.enter()
                    end
                    if self.opt.optim == 'adam' then
                        optim.adam(self:getFeval('G'), self.paramsG, self.optimStateG)
                        if effBatch % self.opt.d_update_freq == 0 then
                            optim.adam(self:getFeval('D'), self.paramsD, self.optimStateD)
                        end
                    else
                        error('Unknown optim method')
                    end
                    effBatch = effBatch + 1
                end
            else
                local errD, dfDo = self:dPass(self.fakePair, REAL_LABEL)
                self.criterionOutG = errD + self.opt.gan_l1_factor * self.criterionG.output
            end
            -- debugger.enter()

            local lossG = PRINT_LOSS_FACTOR * (self.criterionOutG - self.opt.gan_l1_factor * self.criterionG.output)
            local lossD = PRINT_LOSS_FACTOR * self.criterionOutD
            lossGSum = lossGSum + lossG
            lossDSum = lossDSum + lossD
            N = N + 1

            print((' | ' .. set_type .. ' : [%d][%d/%d]     lossL1 %7.3f (%7.3f)   lossG %7.3f (%7.3f)   lossD %7.3f (%7.3f)'):format(
                epoch, batch, batches, lossL1, lossL1Sum / N, lossG, lossGSum / N, lossD, lossDSum / N))

            if doGradUpdate and self.opt.batch_size_factor ~= 1 then
                print('Performed grad update...\n')
            end

            -- if batch % 100 == 0 then
            --     -- Run model on validation set
            --     self:validation(epoch, dataloader)
            --     print('Continuing TRAINING')
            -- end

            -- check that the storage didn't get changed do to an unfortunate getParameters call
            assert(self.paramsG:storage() == self.modelG:parameters()[1]:storage())
            assert(self.paramsD:storage() == self.modelD:parameters()[1]:storage())

            dataTimer:reset()

            if batch_callback then
                batch_callback(x, y, d, output)
            end
        elseif batch > batch_limit then
            break
        else
            print((' | ' .. set_type .. ' : [%d][%d/%d]     === SKIPPED ==='):format(epoch, batch, batches))
        end
    end
    -- self.model:training()

    local lossL1Avg = lossL1Sum / N
    print(('Finished Epoch: [%d]  Time %.3f  Data %.3f  Avg L1 loss: %7.3f\n'):format(
            epoch, timer:time().real, totalDataTime, lossL1Avg))

    return lossL1Avg
end

function Trainer:copyInputs(x, y, d)
    -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
    -- if using DataParallelTable. The target is always copied to a CUDA tensor
    self.input = self.input or torch.CudaTensor()
    self.target = self.target or torch.CudaTensor()
    self.input:resize(x:size()):copy(x)
    self.target:resize(y:size()):copy(y)
end

function Trainer:learningRate(epoch)
    -- Training schedule
    local decay = 0
    if epoch ~= 1 then
        decay = math.floor((epoch - 1) / self.opt.lr_decay_period)
    end
    return self.opt.lr * math.pow(self.opt.lr_decay_factor, decay)
end

return M.GanTrainer
