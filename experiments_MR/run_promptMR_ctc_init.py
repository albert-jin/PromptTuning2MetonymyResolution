from promptMR_ctc_init import experiment_prompt_base


class Parser():
    def __init__(self):
        self.lr = 1e-4
        self.log_file = None
        self.dataset = None
        self.shot_count = None
        self.total_epochs = None
        self.test_pre_epoch = None
        self.lrtemp =None

    def __repr__(self):
        return f'< 学习率: {self.lr}, 数据集: {self.dataset}, Shot: {self.shot_count}, 总轮数: {self.total_epochs}, 测试间隔: {self.test_pre_epoch}, 日志: {self.log_file}>'


if __name__ == '__main__':
    Experiment_Id = 0
    benchmarks = ['ReLocaR', 'CoNLL2003', 'SemEval2007']
    parser = Parser()
    for benchmark in benchmarks:
        # 数据集正常投喂的模型效果.
        parser.dataset = benchmark
        parser.shot_count = 999
        parser.total_epochs = 10
        parser.test_pre_epoch = 1
        parser.lrtemp = 1e-4
        print(f'Experiment Id: {Experiment_Id} ===>', parser)
        experiment_prompt_base(parser)
        # 小样本设置下模型效果
        for shot_num in [5, 10, 20, 50]:
            parser.shot_count = shot_num
            parser.total_epochs = 20
            parser.test_pre_epoch = 2
            experiment_prompt_base(parser)
        print(benchmarks, 'Over.')
    print('All Over.')
