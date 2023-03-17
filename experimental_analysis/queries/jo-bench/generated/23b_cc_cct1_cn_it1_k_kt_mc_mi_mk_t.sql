SELECT * FROM movie_companies AS mc, company_name AS cn, comp_cast_type AS cct1, complete_cast AS cc, kind_type AS kt, title AS t, movie_keyword AS mk, movie_info AS mi, keyword AS k, info_type AS it1 WHERE cct1.kind = 'complete+verified' AND cn.country_code = '[us]' AND it1.info = 'release dates' AND k.keyword IN ('nerd', 'loner', 'alienation', 'dignity') AND kt.kind IN ('movie') AND mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND cct1.id = cc.status_id AND cc.status_id = cct1.id;