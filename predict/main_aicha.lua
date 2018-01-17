require 'paths'
require 'stn'
require 'nn'
nnlib = nn
paths.dofile('util.lua')
paths.dofile('img.lua')
paths.dofile('Get_Alpha.lua')


--------------------------------------------------------------------------------
-- Initialization
--------------------------------------------------------------------------------

if arg[1]  == 'demo' or arg[1] == 'predict-test' then
    cutorch.setDevice(1)
    a = loadAnnotations('aicha-fsrcn-test-0.1/test-bbox')

elseif arg[1] == 'predict-valid' then
    cutorch.setDevice(1)
    a = loadAnnotations('aicha-fsrcn-valid-0.1/test-bbox')--valid set during our exp

else
    print("Please use one of the following input arguments:")
    print("    demo - Generate and display results on the test set")
    print("    predict-valid - Generate predictions on the validation set (MPII images must be available in 'images' directory)")
    print("    predict-test - Generate predictions on the test set")
    return
end

m = torch.load('./final_model_ai-cha-pyranet.t7')

idxs = torch.range(1,a.nsamples)

nsamples = idxs:nElement() 
-- Displays a convenient progress bar
xlua.progress(0,nsamples)
preds = torch.Tensor(nsamples,14,2)
scores = torch.Tensor(nsamples,14,1)

function applyFn(fn, t, t2)
    -- Apply an operation whether passed a table or tensor
    local t_ = {}
    if type(t) == "table" then
        if t2 then
            for i = 1,#t do t_[i] = applyFn(fn, t[i], t2[i]) end
        else
            for i = 1,#t do t_[i] = applyFn(fn, t[i]) end
        end
    else t_ = fn(t, t2) end
    return t_
end


--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------

for i = 1,nsamples do
    -- Set up input image
    local im
    if arg[1] == 'predict-valid' then
        im = image.load('data/valid_images/' .. a['images'][idxs[i]])
    else
        im = image.load('data/test_images/' .. a['images'][idxs[i]])
    end
    
    -- im[1]:add(-0.406)
    -- im[2]:add(-0.457)
    -- im[3]:add(-0.480)

    local imght = im:size()[2]
    local imgwidth = im:size()[3]
    local pt1= torch.Tensor(2)
    local pt2= torch.Tensor(2)
    pt1[1] = a['xmin'][idxs[i]]
    pt1[2] = a['ymin'][idxs[i]]
    pt2[1] = a['xmax'][idxs[i]]
    pt2[2] = a['ymax'][idxs[i]]
    local ht = pt2[2]-pt1[2]
    local width = pt2[1]-pt1[1]
    local scaleRate
    if width > 100 then
        scaleRate = 0.3
    else
        scaleRate = 0.2
    end
    local bias=0
    local rand = torch.rand(1)
    pt1[1] = math.max(0,(pt1[1] - width*scaleRate/2 - rand*width*bias)[1])
    pt1[2] = math.max(0,(pt1[2] - ht*scaleRate/2 - rand*ht*bias)[1])
    pt2[1] = math.max(math.min(imgwidth,(pt2[1] + width*scaleRate/2 + (1-rand)*width*bias)[1]),pt1[1]+5)
    pt2[2] = math.max(math.min(imght,(pt2[2] + ht*scaleRate/2 + (1-rand)*ht*bias)[1]),pt1[2]+5)

    local inputRes = 256
    
    --local inp = crop(im, center, scale, 0, inputRes)
    local inp = cropBox(im, pt1:int(), pt2:int(), 0, inputRes)
    -- Get network output
    local out = m:forward(inp:view(1,3,inputRes,inputRes):cuda())
    -- out = applyFn(function (x) return x:clone() end, out)

    -- for i =1,8 do
    --   out[i] = out[i]:narrow(2,18,16):clone()
    -- end
    -- local flippedOut = m:forward(flip(inp:view(1,3,inputRes,inputRes):cuda()))
    -- for i=1,8 do
    --   flippedOut[i] = flippedOut[i]:narrow(2,18,16):clone()
    -- end
    -- flippedOut = applyFn(function (x) return flip(shuffleLRAIC(x)) end, flippedOut)
    -- flippedOut = applyFn(function (x) return flip(x:clone()) end, flippedOut)
    -- out = applyFn(function (x,y) return x:add(y):div(2) end, out, flippedOut)

    cutorch.synchronize()

    local hm = out[8][1]:float()
    hm[hm:lt(0)] = 0
    local g = image.gaussian(4*1 + 1)
    local s = image.convolve(hm,g,'same')
    hm = s:clone()

    -- Get predictions (hm and img refer to the coordinate space)
    local preds_hm, preds_img, pred_scores = getPreds(hm, pt1:int(), pt2:int())


    preds[i]:copy(preds_img)
    scores[i]:copy(pred_scores)

    xlua.progress(i,nsamples)

    -- Display the result
    if arg[1] == 'demo' then
        preds_hm:mul(inputRes/64) -- Change to input scale
        local dispImg = drawOutput(inp, hm, preds_hm[1])
        w = image.display{image=dispImg,win=w}
        --image.save('preds/images/' .. tostring(i) .. '.jpg',dispImg)
        sys.sleep(3)
    end


    collectgarbage()
end

-- Save predictions
if arg[1] == 'predict-valid' then
    local predFile = hdf5.open('preds/aicha-fsrcn-valid-0.1.h5', 'w')
    predFile:write('preds', preds)
    predFile:write('scores',scores)
    predFile:close()
elseif arg[1] == 'predict-test' then
    local predFile = hdf5.open('preds/aicha-fsrcn-test-0.1.h5', 'w')
    predFile:write('preds', preds)
    predFile:write('scores',scores)
    predFile:close()
elseif arg[1] == 'demo' then
    w.window:close()
end
