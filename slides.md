---
theme: default
highlighter: shiki
transition: slide-left
layout: section
class: 'text-center'
lineNumbers: true
colorSchema: light
---

# 強化学習講義 第六回

シミュレータがあるときの強化学習アルゴリズム

コード：[Google Colab](https://colab.research.google.com/drive/1PHNmbjp6uibp398A9Ds90uVuNiWSUn72?usp=sharing)

用語・表記：[メインスライド](https://syuntoku14.github.io/Foundations-of-RL)を参照

<!--
ほげほげ
-->

---
maxDepth: 1
layout: default
---

<Toc />


---
hideInToc: true
---

## これまでの復習：プランニングアルゴリズム

<br>

<div style="border: 2px solid #000; padding-top: 1px; padding-left: 10px; margin-top: 5px;">

MDPの情報が全部わかっている場合に$\pi^\star$を求める問題を**プランニング問題**と呼ぶ．\
つまり，$(\mathcal{S}, \mathcal{A}, P, r, \mu)$がわかっている場合に$\pi^\star$を**計算する**問題．

これまでの講義で，プランニング問題を解く様々なアルゴリズムを学んできた．\
価値反復法，方策反復法，方策勾配法，線型計画法…

</div>

<br>

<v-click>

<div style="border: 2px solid #000; padding-top: 1px; padding-left: 10px; margin-top: 5px; background-color: #ffffe0;">

しかし，現実の意思決定問題では$P$や$r$がわからないことが多い．

例：歪んだサイコロを振ってみよう．目が出る確率を厳密に近似できるだろうか？

**$P, r$がわからない場合** に$\pi^\star$を求める問題を**強化学習問題**と呼ぶ．

</div>

</v-click>

<span style="border: 2px solid #000; border-radius: 10px; padding: 5px 10px; background-color: #ffffe0; display: inline-block; position: absolute; top: 85%; left: 10%;">
プランニング問題
</span>

<Arrow x1="260" y1="490" x2="400" y2="490" />
<div style="position: absolute; top: 85%; left: 28%; text-align: center; font-size: 0.7em;">
サンプル近似（探索）
</div>

<span style="border: 2px solid #000; border-radius: 10px; padding: 5px 10px; background-color: #ffffe0; display: inline-block; position: absolute; top: 85%; left: 42%;">
強化学習
</span>

<Arrow x1="520" y1="490" x2="670" y2="490" />
<div style="position: absolute; top: 85%; left: 56%; text-align: center; font-size: 0.7em;">
関数近似 （深層）
</div>

<span style="border: 2px solid #000; border-radius: 10px; padding: 5px 10px; background-color: #ffffe0; display: inline-block; position: absolute; top: 85%; left: 70%;">
深層強化学習
</span>

---
hideInToc: true
---

## 今回の内容：シミュレータでの強化学習問題

<br>

**問題設定**

* 今回は簡単のために，$r$は既知だが，$P$がわからない場合を考える<sup>1</sup>．
* 各$(s, a)$について，次の状態$s'$を$P(\cdot \rvert s, a)$からサンプルするシミュレータがあるとする<sup>2</sup>．
* 🤔 シミュレータでの最適方策$\pi^\star$を求めるにはどうすればよいだろうか？ 

<figure style="position: absolute; top: 3%; left: 80%; width: 130px; text-align: center;">
  <img src="./slides/figures/ant-v2.gif" alt="Image description" style="width: 100%;">
  <figcaption style="position: absolute; top: 100%; left: -10%; font-size: 0.8em; word-wrap: break-word; text-align: center; width: 130%">

  [Brax](https://github.com/google/brax)より引用．
  </figcaption>
</figure>

 ---

今回，$P$からサンプルはできるが，$P$の値はわからない．

😢 $P$がわからないので，今までのプランニングアルゴリズムは使えない．

👨‍🏫 これを解決する基本的な強化学習アルゴリズムを紹介するよ


<div style="font-size: 0.7em; text-align: left; position: absolute; bottom: 5px; left: 20px;">

[1] 一般に，$P$がわからない場合の方が難しい．$r$は$|\mathcal{S}||\mathcal{A}|$個しかパラメタがないが，$P$は$|\mathcal{S}|^2|\mathcal{A}|$個のパラメタがある．\
また，$r$は基本的にエンジニアが設計するので，既知な場合が多い．\
[2] このようなシミュレータを，強化学習の理論界隈では**Generative Model**と呼ぶ．
[Reinforcement Learning: Theory and Algorithms](https://rltheorybook.github.io/)など参照．\
最近のシミュレータにはGenerative Modelが備わっている場合がある．（例：[Brax](https://github.com/google/brax) シミュレータ）

</div>

---
src: slides/model-based.md
---

---
src: slides/model-free.md
---

---

## まとめ

今回は「$P$が未知だが，次状態をサンプルするシミュレータがある」ときの強化学習アルゴリズムを学んだ．

* **モデルベース強化学習**：$P$をサンプルで近似してプランニングアルゴリズムを実行する方法．テーブルMDPならば$|\mathcal{S}|^2|\mathcal{A}|$サイズのメモリが必要．
  * 必要なサンプル数は**Hoeffdingの不等式**を使って導出できる（これは次回の探索の話でも使う！）
* **モデルフリー強化学習**：$P$を近似せずに$\pi^\star$を求める方法．テーブルMDPならば$|\mathcal{S}||\mathcal{A}|$サイズのメモリが必要．
今回は価値反復法をサンプルで近似した．性能保証は**誤差伝搬解析**をして示した．

<br>

<div style="border: 2px solid #000; padding-top: 1px; padding-left: 10px; margin-top: 5px; background-color: #ffffe0;">

今回のアルゴリズムはシミュレータを使って次の処理を行っている：

```python
    for (s, a) in product(range(mdp.S), range(mdp.A)):
        next_states = np.random.choice(mdp.S, p=mdp.P[s, a], size=N)  # 各状態行動で，次状態をN個サンプル
```
 
🤔 これが出来ない場合はどうすればよいだろう？現実世界にこうした「リセット機能」はない…

次回は**探索**によって最適方策を求める方法を学ぶ．

</div>