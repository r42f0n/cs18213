java c
CIS 5450 Homework 3: Hypothesis Testing and   Machine   Learning
Due   Date:   October28that   10:00PM   EST
101pointstotal(=85   autograded+   16manuallygraded).
Welcome to CIS 5450 Homework 3! In this homework you will gain some familiarity with   machine learning models for supervised learning. Over the next few days you will strengthen   your   understanding   of   hypothesis   testing   via   simulation   and   ML   concepts   using   baseball,   insurance, and diabetes datasets. Some housekeeping below!   
Before you begin:
·       Be sure to click "Copy to Drive" to make sure you're working on your own personal version   of   the   homework
·       Check the pinned FAQ post on Ed for   updates!   If you have   been   stuck,   chances   are   other   students   have   also   faced   similar   problems.
Note: We will be manually checking your implementations and code for certain problems. If you   incorrectly implemented a procedure using Scikit-learn (e.g. creating predictions on training dataset, incorrectly   process   training   data   prior   to   running   certain   machine   learning   models, hardcoding   values, etc.), we   will   beenforcing   a   penalty   system   up   to   the   maximum   value   of points   allocated   to   the   problem. (e.g. if   your   problem   is   worth   4 points, the   maximum   number   of   points   that   can   be   deducted   is   4 points).
·      Note: If your plot is not run or not   present   after we   open your   notebook, we will deduct the   entire manually graded point value of the plot. (e.g. if your plot is worth 4 points, we will               deduct   4 points).
·      Note: If your   .py   ile is hidden because it's too large, that's ok! We   only   care   about your   .ipynb   ile.
Part 0.   Import and   Setup
Import   necessary   libraries   (do   not   import   anything   else!)
%%capture
!pip3      install      penngrader-client
import      numpy      as      np
import      pandas      as      pd
import      seaborn      as      sns
import      matplotlib.pyplot      as      plt
from      sklearn.neighbors      import      KNeighborsClassifier      from      sklearn.linear_model      import      LogisticRegression
from      sklearn.model_selection      import      train_test_split,      GridSearchCV,      StratifiedKFold
from      sklearn.metrics      import      accuracy_score,      precision_score,      recall_score
from      sklearn.preprocessing      import      OneHotEncoder,      OrdinalEncoder,      StandardScaler
from      sklearn.ensemble      import      RandomForestClassifier
import      random
import      math
from      xgboost      import      XGBClassifier   from      penngrader.grader      import      *
!apt      install      zstd
!wget      -nc      -O      diabetes_prediction_dataset.csv.zst      https://www.dropbox.com/scl/fi/p8qpv4eja0xp3
!unzstd      -f      diabetes_prediction_dataset.csv.zst
!wget      -nc      -O      games.csv.zst      https://www.dropbox.com/scl/fi/43au9nv0bty84pqg6aw64/games.csv.zst
!unzstd      -f      games.csv.zst
!wget      -nc      -O      medical_cost.csv.zst      https://www.dropbox.com/scl/fi/8nz07htxxi07xilddsulx/medica
!unzstd      -f      medical_cost.csv.zst
PennGrader Setup
#      PLEASE      ENSURE      YOUR      PENN-ID      IS      ENTERED      CORRECTLY.      IF      NOT,      THE      AUTOGRADER      WON'T      KNOW
#      TO      ASSIGN      POINTS      TO      YOU      IN      OUR      BACKEND
STUDENT_ID      =            #      YOUR      PENN-ID      GOES      HERE      AS      AN      INTEGER      #
SECRET      =      STUDENT_ID
%%writefile      config.yaml


grader_api_url:      'https://23whrwph9h.execute-api.us-east-1.amazonaw   s.com/default/Grader23
d                     k               '   l   k                                                            k                                             '
%set_env      HW_ID=cis5450_fall24_HW3
grader      =      PennGrader('config.yaml',      'cis5450_fall24_HW3',      STUDENT_ID,      SECRET)
Part 1: Hypothesis Testing via Simulation [17   Points Total]
1.1: Estimating   Pi   through   Simulation   [4 points]
Consider a circle with radius 1/2 inside of a   unit   square:
   
We could compute the area of the circle with a well-known formula and the value of π, but we   can also compute both the area of the circle, and the mysterious value of π, via simulation!
If   we   randomly   sample   a   point   inside   the   unit   square, the   probability   that   the   point   falls   within
the circle is equal to the area of the circle divided by the area of the square. Thus, if we sample a   total of Pt   points and Pc   of them are in the circle, we can write the area of the circle Ac   as:
   
Solving for π gives:
   Below   is   some   Python   code   that   simulates   picking   a   random   point   in   the   square, testing   if   that point is inside the circle, and keeping track of Pc    and Pt. Run this code to ensure it works, and   see   how   long   it   takes. The   simulation   should   sample   10 million   points.


%%time
def      pt_in_circle(x,      y):
return      math.sqrt(x**2      +      y**2)      <      0.5
for      i      in      range   (10_000_000):
#      Sample      x      and      y      uniformly      from      -0.5      to      0.5
x      =      random.uniform(-0.5,      0.5)
y      =      random.uniform(-0.5,      0.5)
if      pt_in_circle(x,      y):
p_c      +=      1
p_t      +=      1
#      Estimate      pi
pi_estimate      =      4      *      p_c      /      p_t
print(f"Estimated      value      of      pi:      {pi_estimate}")
Next, let's accelerate our simulation with vectorization! Using    numpy   , write a vectorized version of   the   simulation.
Your   solution   must:
·       Contain no loops (   while   ,    for   , etc.) or list comprehensions
·      Contain   no   if   statements   or   conditionals
·       Only use built-in or numpy    np.   functions
·       Should make only a single   call to    np.random   .
Note:You   will   not   get   any   credit   if   you   violate   any   of   the   above   instructions.
%%time
#      TODO:      Sample      10,000,000      points      and      calculate      pi
pi_estimate      =
#      Do      NOT      change      anything      below      this      line
grader.grade(test_case_id      =      'test_estimate_pi',      answer      =      (pi_estimate,      _ih[-1]))
1.2 Hypothesis Testing [13   Points]
It 代 写CIS 5450 Homework 3: Hypothesis Testing and Machine LearningPython
代做程序编程语言is commonly believed that in many sports, the home team tends to have an advantage over   the   away   team, often   referred   to   as   "home   ield   advantage." In   this   part, we   will   perform   a   hypothesis   test   to   determine, from   a   statistical   standpoint, if   such   an   advantage   exists   in   Major   League Baseball (MLB) games. We will guide you through each step of the process, using the MLB Games Dataset, and conduct the test through simulation. For this part, we will be using   vectorization only, so no    for   loops should be used!
1.2.1   Load   Data
Before   diving   into   the   simulation, we   need   to   load   in   the   data   i   rst.
TODO:
·       Load    games.csv   and save the   data to a dataframe. called    games_df   .
·         Inspect   theirst   ive   rows. There   are   many   columns   in   this   dataframe, but   think   about   which   ones   we   will   actually   need   for   hypothesis   testing.
#      DO      NOT      CHANGE   #      Import      Data
games_df      =      pd.read_csv("games.csv")   games_df.head()
In lecture, you have learned that in hypothesis testing, we start by assuming   a   baseline called the   null   hypothesis. For   this   test, the   null   hypothesis   (H0   ) is   that   home   ield   advantage   does   not   exist in MLB games. This means that, under the   null hypothesis, the probability of the home   team winning is equal to the probability of the away team winning (i.e., 50%).
As a brief review, to determine whether we can reject or fail to reject the null hypothesis, we will:
1. Set up the hypotheses:
o         H0: The   probability   of   the   home   team   winning   is   50% (no   home   ield   advantage).
o         Ha      (alternative   hypothesis): The   probability   of   the   home   team   winning   is   greater   than   50% (home   ield   advantage   exists).
2. Analyze the data: We will calculate the observed proportion of home team wins using the   MLB Games   Dataset.
3. Simulate random outcomes: We will simulate a large number of seasons where home
teams win exactly 50% of the time, assuming the null hypothesis is true. In lecture you saw   the Gaussian distribution. Here, the nature of the data requires us to draw from the
binomial distribution.
4. Compare   observed   results   to   simulations: We   will   determine   how   often   the   simulated
results produce home win rates as   extreme or more extreme than the observed data. This   will give us a p-value, which tells us the likelihood of observing the current data under the   null   hypothesis.
5. Make a decision: Based on the p-value, we will decide whether to reject or fail to reject the   null hypothesis:
o         If the p-value is below a threshold (in this case we'll use 0.05), we will reject the null   hypothesis   and   conclude   that   home   ield   advantage   likely   exists. Intuitively, a   small   p-value means that the data we observed is extremely unlikely to occur under the null   hypothesis.
。   If the p-value is higher, we will fail to reject the null hypothesis, meaning the evidence   is   not   strong   enough   to   suggest   home   ield   advantage.
1.2.2 Calculate Original Test Statistic [3 Points]
We will now move on to Step 2, which is to calculate the original test   statistic, i.e., the home win   rate of the given data.
TODO:
·      Under special circumstances, there can be a tie. For this part,   just drop games that ended   in   a tie.
·       Create a column called    home_win   that is a 1 if the home team won that game and 0   otherwise.
·       Calculate   the   proportion   of   times   that   the   home   team   wins   and   store   it   in   home_win_rate
#      Drop      ties      and      reset      index
#      Create      a      column      that      is      1      if      the      home      team      won
#      Calculate      original      test      statistic
home_win_rate      =
#      Grader      Cell      (3      points)
grader.grade(test_case_id      =      'original_test_statistic',      answer      =      (games_df['home_win'],      home
1.2.3 Simulation and Plotting [7 points] (7   manually   graded)
Now   we   will   simulate   the   null   world   to   get   a   distribution.
TODO:
·      Simulate 10,000,000   trials
·       Each   trial   should   be   drawn   from   a   binomial   distribution. If   you   are   unfamiliar   with   the
binomial   distribution, take   a   look   at   the   numpy   documentation   and   pay   careful   attention   to   what your   n   and   p   should be in this case. Recall what it   means when we are sampling
from   the   null   world.
·      Calculate   the   simulated   proportion   by   dividing   by   total   games Note:You should be using    numpy   vectorization
#      Simulate      random      home      win      outcomes      for      each      game      across      all      simulation
simulated_home_wins      =


#      Calculate      the      simulated      proportion      of      home      wins
l         d
Now, let's   visualize   the   distribution   of   our   simulation.
Task:
·      Plot   a   histogram   of   the   distribution   of   the   test   statistic   in   the   null   world. Use   50 bins.
·      Title the plot: Distribution   of   Simulated   Home   Winrate
·       Label the x-axis: Home   Winrate ·       Label they-axis: Frequency
·      Add   a   red   vertical   line   to   indicate   the   original   test   statistic. Label   this   vertical   line "Observed Home Winrate: {home_win_rate}". Round to four decimal points.
·      Add a legend to your plot.
Hint:Take a look at   matplotlib.pyplot.axvline
#      Plot      the      distribution      of      simulated      home      win      counts
1.2.4 Calculate p-value [3 points]   (2   manually graded)
Finally, we can calculate the simulated p-value. Remember what the p-value represents, and   use   your   simulated   win   rates   to   calculate   it.
#      Calculate      the      p-value
simulated   p   value      =
#      Grader      Cell      (1      points)
grader.grade(test_case_id      =      'test   p   value',      answer      =      simulated   p   value)
After calculating the p-value, briely describe what it represents. Does the p-value represent
exactly   what   you   might   get   if   you   were   to   calculate   it   mathematically? State   whether   we   should reject   or   fail   to   reject   the   null   hypothesis.
   





         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
