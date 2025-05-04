Q:
1. 白底背景和黑背景或者随机背景到底有什么区别？我看到在稠密化那里针对白底背景单独做了一次透明度重置，为什么？为什么黑底或者其它底就不用？

TODO:
1. GaussianModel 这个类的相关实现要好好琢磨, scene.gaussian_model.GaussianModel
    1) self.xyz_scheduler_args = get_expon_lr_func() 这里有问题感觉, lr_delay_mult 传了跟没传一样, 因为没传 lr_delay_steps
    2) 在 densify_and_clone, densify_and_split 之后 self.max_radii2D 全是 0 吧, 那后面根据 self.max_radii2D 去剪枝的逻辑还有什么存在的意义
3. render 函数是需要好好去研读的, gaussian_renderer.render
4. LangSplat 这篇 paper 似乎跟我要实现的功能 3 比较接近, 到时候可以去参考其开源代码
