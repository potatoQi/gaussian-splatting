Q:
1. 白底背景和黑背景或者随机背景到底有什么区别？我看到在稠密化那里针对白底背景单独做了一次透明度重置，为什么？为什么黑底或者其它底就不用？
2. model save 跟 ckpt save 的区别是什么？model save 是 scene.save(iteration), ckpt save 是 torch.save((gaussians.capture(), iteration),scene.model_path + "/chkpnt" + str(iteration) + ".pth"). 目前我的理解是 ckpt save 时的 gaussians.capture() 就是把最后一次 model save 的结果保存下来。不过这样的话不是很难受吗？ckpt save 出来的 model 跟当前的 iteration 不挂钩, 反而跟最后一次 model save 的结果挂钩

TODO:
1. GaussianModel 这个类的相关实现要好好琢磨, scene.gaussian_model.GaussianModel
2. Scene 这个类的相关实现要好好琢磨, scene.Scene
3. render 函数是需要好好去研读的, gaussian_renderer.render
4. LangSplat 这篇 paper 似乎跟我要实现的功能 3 比较接近, 到时候可以去参考其开源代码