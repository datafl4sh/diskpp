#pragma once

#include <vector>

#include "diskpp/common/simplicial_formula.hpp"
#include "diskpp/quadratures/quadrature_point.hpp"

namespace disk::quadrature {

namespace priv {

struct dunavant_point {
    size_t      perms;
    double      coords[3];
    double      weight;
};

static dunavant_point dunavant_rule_1[] = {
    { 1, { 0.333333333333333, 0.333333333333333, 0.333333333333333 },  1.000000000000000 }
};

static dunavant_point dunavant_rule_2[] = {
    { 3, { 0.666666666666667, 0.166666666666667, 0.166666666666667 },  0.333333333333333 }
};

static dunavant_point dunavant_rule_3[] = {
    { 1, { 0.333333333333333, 0.333333333333333, 0.333333333333333 }, -0.562500000000000 },
    { 3, { 0.600000000000000, 0.200000000000000, 0.200000000000000 },  0.520833333333333 }
};

static dunavant_point dunavant_rule_4[] = {
    { 3, { 0.108103018168070, 0.445948490915965, 0.445948490915965 },  0.223381589678011 },
    { 3, { 0.816847572980459, 0.091576213509771, 0.091576213509771 },  0.109951743655322 }
};

static dunavant_point dunavant_rule_5[] = {
    { 1, { 0.333333333333333, 0.333333333333333, 0.333333333333333 },  0.225000000000000 },
    { 3, { 0.059715871789770, 0.470142064105115, 0.470142064105115 },  0.132394152788506 },
    { 3, { 0.797426985353087, 0.101286507323456, 0.101286507323456 },  0.125939180544827 }
};

static dunavant_point dunavant_rule_6[] = {
    { 3, { 0.501426509658179, 0.249286745170910, 0.249286745170910 },  0.116786275726379 },
    { 3, { 0.873821971016996, 0.063089014491502, 0.063089014491502 },  0.050844906370207 },
    { 6, { 0.053145049844817, 0.310352451033784, 0.636502499121399 },  0.082851075618374 }
};

static dunavant_point dunavant_rule_7[] = {
    { 1, { 0.333333333333333, 0.333333333333333, 0.333333333333333 }, -0.149570044467682 },
    { 3, { 0.479308067841920, 0.260345966079040, 0.260345966079040 },  0.175615257433208 },
    { 3, { 0.869739794195568, 0.065130102902216, 0.065130102902216 },  0.053347235608838 },
    { 6, { 0.048690315425316, 0.312865496004874, 0.638444188569810 },  0.077113760890257 },
};

static dunavant_point dunavant_rule_8[] = {
    { 1, { 0.333333333333333, 0.333333333333333, 0.333333333333333 },  0.144315607677787 },
    { 3, { 0.081414823414554, 0.459292588292723, 0.459292588292723 },  0.095091634267285 },
    { 3, { 0.658861384496480, 0.170569307751760, 0.170569307751760 },  0.103217370534718 },
    { 3, { 0.898905543365938, 0.050547228317031, 0.050547228317031 },  0.032458497623198 },
    { 6, { 0.008394777409958, 0.263112829634638, 0.728492392955404 },  0.027230314174435 }
};

static dunavant_point dunavant_rule_9[] = {
    { 1, { 0.333333333333333, 0.333333333333333, 0.333333333333333 },  0.097135796282799 },
    { 3, { 0.020634961602525, 0.489682519198738, 0.489682519198738 },  0.031334700227139 },
    { 3, { 0.125820817014127, 0.437089591492937, 0.437089591492937 },  0.077827541004774 },
    { 3, { 0.623592928761935, 0.188203535619033, 0.188203535619033 },  0.079647738927210 },
    { 3, { 0.910540973211095, 0.044729513394453, 0.044729513394453 },  0.025577675658698 },
    { 6, { 0.036838412054736, 0.221962989160766, 0.741198598784498 },  0.043283539377289 }
};

static dunavant_point dunavant_rule_10[] = {
    { 1, { 0.333333333333333, 0.333333333333333, 0.333333333333333 },  0.090817990382754 },
    { 3, { 0.028844733232685, 0.485577633383657, 0.485577633383657 },  0.036725957756467 },
    { 3, { 0.781036849029926, 0.109481575485037, 0.109481575485037 },  0.045321059435528 },
    { 6, { 0.141707219414880, 0.307939838764121, 0.550352941820999 },  0.072757916845420 },
    { 6, { 0.025003534762686, 0.246672560639903, 0.728323904597411 },  0.028327242531057 },
    { 6, { 0.009540815400299, 0.066803251012200, 0.923655933587500 },  0.009421666963733 }
};

static dunavant_point dunavant_rule_11[] = {
    { 3, {-0.069222096541517, 0.534611048270758, 0.534611048270758 },  0.000927006328961 },
    { 3, { 0.202061394068290, 0.398969302965855, 0.398969302965855 },  0.077149534914813 },
    { 3, { 0.593380199137435, 0.203309900431282, 0.203309900431282 },  0.059322977380774 },
    { 3, { 0.761298175434837, 0.119350912282581, 0.119350912282581 },  0.036184540503418 },
    { 3, { 0.935270103777448, 0.032364948111276, 0.032364948111276 },  0.013659731002678 },
    { 6, { 0.050178138310495, 0.356620648261293, 0.593201213428213 },  0.052337111962204 },
    { 6, { 0.021022016536166, 0.171488980304042, 0.807489003159792 },  0.020707659639141 }
};

static dunavant_point dunavant_rule_12[] = {
    { 3, { 0.023565220452390, 0.488217389773805, 0.488217389773805 },  0.025731066440455 },
    { 3, { 0.120551215411079, 0.439724392294460, 0.439724392294460 },  0.043692544538038 },
    { 3, { 0.457579229975768, 0.271210385012116, 0.271210385012116 },  0.062858224217885 },
    { 3, { 0.744847708916828, 0.127576145541586, 0.127576145541586 },  0.034796112930709 },
    { 3, { 0.957365299093579, 0.021317350453210, 0.021317350453210 },  0.006166261051559 },
    { 6, { 0.115343494534698, 0.275713269685514, 0.608943235779788 },  0.040371557766381 },
    { 6, { 0.022838332222257, 0.281325580989940, 0.695836086787803 },  0.022356773202303 },
    { 6, { 0.025734050548330, 0.116251915907597, 0.858014033544073 },  0.017316231108659 }
};

static dunavant_point dunavant_rule_13[] = {
    { 1, { 0.333333333333333, 0.333333333333333, 0.333333333333333 },  0.052520923400802 },
    { 3, { 0.009903630120591, 0.495048184939705, 0.495048184939705 },  0.011280145209330 },
    { 3, { 0.062566729780852, 0.468716635109574, 0.468716635109574 },  0.031423518362454 },
    { 3, { 0.170957326397447, 0.414521336801277, 0.414521336801277 },  0.047072502504194 },
    { 3, { 0.541200855914337, 0.229399572042831, 0.229399572042831 },  0.047363586536355 },
    { 3, { 0.771151009607340, 0.114424495196330, 0.114424495196330 },  0.031167529045794 },
    { 3, { 0.950377217273082, 0.024811391363459, 0.024811391363459 },  0.007975771465074 },
    { 6, { 0.094853828379579, 0.268794997058761, 0.636351174561660 },  0.036848402728732 },
    { 6, { 0.018100773278807, 0.291730066734288, 0.690169159986905 },  0.017401463303822 },
    { 6, { 0.022233076674090, 0.126357385491669, 0.851409537834241 },  0.015521786839045 }
};

static dunavant_point dunavant_rule_14[] = {
    { 3, { 0.022072179275643, 0.488963910362179, 0.488963910362179 },  0.021883581369429 },
    { 3, { 0.164710561319092, 0.417644719340454, 0.417644719340454 },  0.032788353544125 },
    { 3, { 0.453044943382323, 0.273477528308839, 0.273477528308839 },  0.051774104507292 },
    { 3, { 0.645588935174913, 0.177205532412543, 0.177205532412543 },  0.042162588736993 },
    { 3, { 0.876400233818255, 0.061799883090873, 0.061799883090873 },  0.014433699669777 },
    { 3, { 0.961218077502598, 0.019390961248701, 0.019390961248701 },  0.004923403602400 },
    { 6, { 0.057124757403648, 0.172266687821356, 0.770608554774996 },  0.024665753212564 },
    { 6, { 0.092916249356972, 0.336861459796345, 0.570222290846683 },  0.038571510787061 },
    { 6, { 0.014646950055654, 0.298372882136258, 0.686980167808088 },  0.014436308113534 },
    { 6, { 0.001268330932872, 0.118974497696957, 0.879757171370171 },  0.005010228838501 }
};

static dunavant_point dunavant_rule_15[] = {
    { 3, {-0.013945833716486, 0.506972916858243, 0.506972916858243 },  0.001916875642849 },
    { 3, { 0.137187291433955, 0.431406354283023, 0.431406354283023 },  0.044249027271145 },
    { 3, { 0.444612710305711, 0.277693644847144, 0.277693644847144 },  0.051186548718852 },
    { 3, { 0.747070217917492, 0.126464891041254, 0.126464891041254 },  0.023687735870688 },
    { 3, { 0.858383228050628, 0.070808385974686, 0.070808385974686 },  0.013289775690021 },
    { 3, { 0.962069659517853, 0.018965170241073, 0.018965170241073 },  0.004748916608192 },
    { 6, { 0.133734161966621, 0.261311371140087, 0.604954466893291 },  0.038550072599593 },
    { 6, { 0.036366677396917, 0.388046767090269, 0.575586555512814 },  0.027215814320624 },
    { 6, {-0.010174883126571, 0.285712220049916, 0.724462663076655 },  0.002182077366797 },
    { 6, { 0.036843869875878, 0.215599664072284, 0.747556466051838 },  0.021505319847731 },
    { 6, { 0.012459809331199, 0.103575616576386, 0.883964574092416 },  0.007673942631049 }
};

static dunavant_point dunavant_rule_16[] = {
    { 1, { 0.333333333333333, 0.333333333333333, 0.333333333333333 },  0.046875697427642 },
    { 3, { 0.005238916103123, 0.497380541948438, 0.497380541948438 },  0.006405878578585 },
    { 3, { 0.173061122901295, 0.413469438549352, 0.413469438549352 },  0.041710296739387 },
    { 3, { 0.059082801866017, 0.470458599066991, 0.470458599066991 },  0.026891484250064 },
    { 3, { 0.518892500060958, 0.240553749969521, 0.240553749969521 },  0.042132522761650 },
    { 3, { 0.704068411554854, 0.147965794222573, 0.147965794222573 },  0.030000266842773 },
    { 3, { 0.849069624685052, 0.075465187657474, 0.075465187657474 },  0.014200098925024 },
    { 3, { 0.966807194753950, 0.016596402623025, 0.016596402623025 },  0.003582462351273 },
    { 6, { 0.103575692245252, 0.296555596579887, 0.599868711174861 },  0.032773147460627 },
    { 6, { 0.020083411655416, 0.337723063403079, 0.642193524941505 },  0.015298306248441 },
    { 6, {-0.004341002614139, 0.204748281642812, 0.799592720971327 },  0.002386244192839 },
    { 6, { 0.041941786468010, 0.189358492130623, 0.768699721401368 },  0.019084792755899 },
    { 6, { 0.014317320230681, 0.085283615682657, 0.900399064086661 },  0.006850054546542 }
};

static dunavant_point dunavant_rule_17[] = {
    { 1, { 0.333333333333333, 0.333333333333333, 0.333333333333333 },  0.033437199290803 },
    { 3, { 0.005658918886452, 0.497170540556774, 0.497170540556774 },  0.005093415440507 },
    { 3, { 0.035647354750751, 0.482176322624625, 0.482176322624625 },  0.014670864527638 },
    { 3, { 0.099520061958437, 0.450239969020782, 0.450239969020782 },  0.024350878353672 },
    { 3, { 0.199467521245206, 0.400266239377397, 0.400266239377397 },  0.031107550868969 },
    { 3, { 0.495717464058095, 0.252141267970953, 0.252141267970953 },  0.031257111218620 },
    { 3, { 0.675905990683077, 0.162047004658461, 0.162047004658461 },  0.024815654339665 },
    { 3, { 0.848248235478508, 0.075875882260746, 0.075875882260746 },  0.014056073070557 },
    { 3, { 0.968690546064356, 0.015654726967822, 0.015654726967822 },  0.003194676173779 },
    { 6, { 0.010186928826919, 0.334319867363658, 0.655493203809423 },  0.008119655318993 },
    { 6, { 0.135440871671036, 0.292221537796944, 0.572337590532020 },  0.026805742283163 },
    { 6, { 0.054423924290583, 0.319574885423190, 0.626001190286228 },  0.018459993210822 },
    { 6, { 0.012868560833637, 0.190704224192292, 0.796427214974071 },  0.008476868534328 },
    { 6, { 0.067165782413524, 0.180483211648746, 0.752351005937729 },  0.018292796770025 },
    { 6, { 0.014663182224828, 0.080711313679564, 0.904625504095608 },  0.006665632004165 }
};

static dunavant_point dunavant_rule_18[] = {
    { 1, { 0.333333333333333, 0.333333333333333, 0.333333333333333 },  0.030809939937647 },
    { 3, { 0.013310382738157, 0.493344808630921, 0.493344808630921 },  0.009072436679404 },
    { 3, { 0.061578811516086, 0.469210594241957, 0.469210594241957 },  0.018761316939594 },
    { 3, { 0.127437208225989, 0.436281395887006, 0.436281395887006 },  0.019441097985477 },
    { 3, { 0.210307658653168, 0.394846170673416, 0.394846170673416 },  0.027753948610810 },
    { 3, { 0.500410862393686, 0.249794568803157, 0.249794568803157 },  0.032256225351457 },
    { 3, { 0.677135612512315, 0.161432193743843, 0.161432193743843 },  0.025074032616922 },
    { 3, { 0.846803545029257, 0.076598227485371, 0.076598227485371 },  0.015271927971832 },
    { 3, { 0.951495121293100, 0.024252439353450, 0.024252439353450 },  0.006793922022963 },
    { 3, { 0.913707265566071, 0.043146367216965, 0.043146367216965 }, -0.002223098729920 },
    { 6, { 0.008430536202420, 0.358911494940944, 0.632657968856636 },  0.006331914076406 },
    { 6, { 0.131186551737188, 0.294402476751957, 0.574410971510855 },  0.027257538049138 },
    { 6, { 0.050203151565675, 0.325017801641814, 0.624779046792512 },  0.017676785649465 },
    { 6, { 0.066329263810916, 0.184737559666046, 0.748933176523037 },  0.018379484638070 },
    { 6, { 0.011996194566236, 0.218796800013321, 0.769207005420443 },  0.008104732808192 },
    { 6, { 0.014858100590125, 0.101179597136408, 0.883962302273467 },  0.007634129070725 },
    { 6, {-0.035222015287949, 0.020874755282586, 1.014347260005363 },  0.000046187660794 }
};

static dunavant_point dunavant_rule_19[] = {
    { 1, { 0.333333333333333, 0.333333333333333, 0.333333333333333 },  0.032906331388919 },
    { 3, { 0.020780025853987, 0.489609987073006, 0.489609987073006 },  0.010330731891272 },
    { 3, { 0.090926214604215, 0.454536892697893, 0.454536892697893 },  0.022387247263016 },
    { 3, { 0.197166638701138, 0.401416680649431, 0.401416680649431 },  0.030266125869468 },
    { 3, { 0.488896691193805, 0.255551654403098, 0.255551654403098 },  0.030490967802198 },
    { 3, { 0.645844115695741, 0.177077942152130, 0.177077942152130 },  0.024159212741641 },
    { 3, { 0.779877893544096, 0.110061053227952, 0.110061053227952 },  0.016050803586801 },
    { 3, { 0.888942751496321, 0.055528624251840, 0.055528624251840 },  0.008084580261784 },
    { 3, { 0.974756272445543, 0.012621863777229, 0.012621863777229 },  0.002079362027485 },
    { 6, { 0.003611417848412, 0.395754787356943, 0.600633794794645 },  0.003884876904981 },
    { 6, { 0.134466754530780, 0.307929983880436, 0.557603261588784 },  0.025574160612022 },
    { 6, { 0.014446025776115, 0.264566948406520, 0.720987025817365 },  0.008880903573338 },
    { 6, { 0.046933578838178, 0.358539352205951, 0.594527068955871 },  0.016124546761731 },
    { 6, { 0.002861120350567, 0.157807405968595, 0.839331473680839 },  0.002491941817491 },
    { 6, { 0.223861424097916, 0.075050596975911, 0.701087978926173 },  0.018242840118951 },
    { 6, { 0.034647074816760, 0.142421601113383, 0.822931324069857 },  0.010258563736199 },
    { 6, { 0.010161119296278, 0.065494628082938, 0.924344252620784 },  0.003799928855302 }
};

static dunavant_point dunavant_rule_20[] = {
    { 1, { 0.333333333333333, 0.333333333333333, 0.333333333333333 }, 0.033057055541624 },
    { 3, {-0.001900928704400, 0.500950464352200, 0.500950464352200 }, 0.000867019185663 },
    { 3, { 0.023574084130543, 0.488212957934729, 0.488212957934729 }, 0.011660052716448 },
    { 3, { 0.089726636099435, 0.455136681950283, 0.455136681950283 }, 0.022876936356421 },
    { 3, { 0.196007481363421, 0.401996259318289, 0.401996259318289 }, 0.030448982673938 },
    { 3, { 0.488214180481157, 0.255892909759421, 0.255892909759421 }, 0.030624891725355 },
    { 3, { 0.647023488009788, 0.176488255995106, 0.176488255995106 }, 0.024368057676800 },
    { 3, { 0.791658289326483, 0.104170855336758, 0.104170855336758 }, 0.015997432032024 },
    { 3, { 0.893862072318140, 0.053068963840930, 0.053068963840930 }, 0.007698301815602 },
    { 3, { 0.916762569607942, 0.041618715196029, 0.041618715196029 },-0.000632060497488 },
    { 3, { 0.976836157186356, 0.011581921406822, 0.011581921406822 }, 0.001751134301193 },
    { 6, { 0.048741583664839, 0.344855770229001, 0.606402646106160 }, 0.016465839189576 },
    { 6, { 0.006314115948605, 0.377843269594854, 0.615842614456541 }, 0.004839033540485 },
    { 6, { 0.134316520547348, 0.306635479062357, 0.559048000390295 }, 0.025804906534650 },
    { 6, { 0.013973893962392, 0.249419362774742, 0.736606743262866 }, 0.008471091054441 },
    { 6, { 0.075549132909764, 0.212775724802802, 0.711675142287434 }, 0.018354914106280 },
    { 6, {-0.008368153208227, 0.146965436053239, 0.861402717154987 }, 0.000704404677908 },
    { 6, { 0.026686063258714, 0.137726978828923, 0.835586957912363 }, 0.010112684927462 },
    { 6, { 0.010547719294141, 0.059696109149007, 0.929756171556853 }, 0.003573909385950 }
};

struct dunavant_rule {
    size_t          num_entries;
    size_t          num_points;
    dunavant_point  *points;
};

static struct dunavant_rule dunavant_rules[] = {
    {  sizeof(dunavant_rule_1)/(sizeof(dunavant_point)),  1, dunavant_rule_1  },
    {  sizeof(dunavant_rule_2)/(sizeof(dunavant_point)),  3, dunavant_rule_2  },
    {  sizeof(dunavant_rule_3)/(sizeof(dunavant_point)),  4, dunavant_rule_3  },
    {  sizeof(dunavant_rule_4)/(sizeof(dunavant_point)),  6, dunavant_rule_4  },
    {  sizeof(dunavant_rule_5)/(sizeof(dunavant_point)),  7, dunavant_rule_5  },
    {  sizeof(dunavant_rule_6)/(sizeof(dunavant_point)), 12, dunavant_rule_6  },
    {  sizeof(dunavant_rule_7)/(sizeof(dunavant_point)), 13, dunavant_rule_7  },
    {  sizeof(dunavant_rule_8)/(sizeof(dunavant_point)), 16, dunavant_rule_8  },
    {  sizeof(dunavant_rule_9)/(sizeof(dunavant_point)), 19, dunavant_rule_9  },
    { sizeof(dunavant_rule_10)/(sizeof(dunavant_point)), 25, dunavant_rule_10 },
    { sizeof(dunavant_rule_11)/(sizeof(dunavant_point)), 27, dunavant_rule_11 },
    { sizeof(dunavant_rule_12)/(sizeof(dunavant_point)), 33, dunavant_rule_12 },
    { sizeof(dunavant_rule_13)/(sizeof(dunavant_point)), 37, dunavant_rule_13 },
    { sizeof(dunavant_rule_14)/(sizeof(dunavant_point)), 42, dunavant_rule_14 },
    { sizeof(dunavant_rule_15)/(sizeof(dunavant_point)), 48, dunavant_rule_15 },
    { sizeof(dunavant_rule_16)/(sizeof(dunavant_point)), 52, dunavant_rule_16 },
    { sizeof(dunavant_rule_17)/(sizeof(dunavant_point)), 61, dunavant_rule_17 },
    { sizeof(dunavant_rule_18)/(sizeof(dunavant_point)), 70, dunavant_rule_18 },
    { sizeof(dunavant_rule_19)/(sizeof(dunavant_point)), 73, dunavant_rule_19 },
    { sizeof(dunavant_rule_20)/(sizeof(dunavant_point)), 79, dunavant_rule_20 },
    { 0, 0, NULL }
};

#define QUAD_RAW_DUNAVANT_MAX_ORDER 20

} // namespace priv

template<typename T, size_t DIM>
std::vector<quadrature_point<T,DIM>>
dunavant(size_t degree, const point<T,DIM>& p0, const point<T,DIM>& p1, const point<T,DIM>& p2)
{
    auto max_degree = sizeof(priv::dunavant_rules)/sizeof(priv::dunavant_rule);
    if (degree > max_degree)
        throw std::invalid_argument("Dunavant quadrature: degree too high");

    size_t rule_num = (degree == 0) ? 0 : degree - 1;
    assert(rule_num < max_degree-1);

    auto area = area_triangle_kahan(p0, p1, p2);

    auto num_entries = priv::dunavant_rules[rule_num].num_entries;
    auto num_points = priv::dunavant_rules[rule_num].num_points;

    std::vector<quadrature_point<T,DIM>> ret;
    ret.reserve(num_points);

    for (size_t i = 0; i < num_entries; i++)
    {
        auto& dp = priv::dunavant_rules[rule_num].points[i];

        auto l0 = dp.coords[0];
        auto l1 = dp.coords[1];
        auto l2 = dp.coords[2];
        auto w = dp.weight;

        switch (dp.perms)
        {
            case 1:
                ret.push_back({p0*l0 + p1*l1 + p2*l2, w*area});
                break;

            case 3:
                ret.push_back({p0*l0 + p1*l1 + p2*l2, w*area});
                ret.push_back({p0*l1 + p1*l2 + p2*l0, w*area});
                ret.push_back({p0*l1 + p1*l0 + p2*l2, w*area});
                break;

            case 6:
                ret.push_back({p0*l0 + p1*l1 + p2*l2, w*area});
                ret.push_back({p0*l0 + p1*l2 + p2*l1, w*area});
                ret.push_back({p0*l1 + p1*l0 + p2*l2, w*area});
                ret.push_back({p0*l1 + p1*l2 + p2*l0, w*area});
                ret.push_back({p0*l2 + p1*l0 + p2*l1, w*area});
                ret.push_back({p0*l2 + p1*l1 + p2*l0, w*area});
                break;
        }
    }

    return ret;
}

template<typename T, size_t DIM>
std::vector<quadrature_point<T,DIM>>
dunavant(size_t degree, const std::array<point<T,DIM>, 3>& pts)
{
    return dunavant(degree, pts[0], pts[1], pts[2]);
}

} // namespace disk::quadrature
