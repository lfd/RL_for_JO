SELECT * FROM keyword AS k, cast_info AS ci, movie_keyword AS mk, title AS t, movie_companies AS mc, company_name AS cn, complete_cast AS cc WHERE ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND k.keyword = 'computer-animation' AND t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2005 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;