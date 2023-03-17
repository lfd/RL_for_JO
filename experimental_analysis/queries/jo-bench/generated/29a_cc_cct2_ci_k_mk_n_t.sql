SELECT * FROM name AS n, title AS t, keyword AS k, complete_cast AS cc, movie_keyword AS mk, comp_cast_type AS cct2, cast_info AS ci WHERE cct2.kind = 'complete+verified' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND k.keyword = 'computer-animation' AND n.gender = 'f' AND n.name LIKE '%An%' AND t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;