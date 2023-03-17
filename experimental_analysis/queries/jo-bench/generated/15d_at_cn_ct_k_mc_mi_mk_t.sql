SELECT * FROM movie_keyword AS mk, title AS t, aka_title AS at, keyword AS k, movie_companies AS mc, company_type AS ct, company_name AS cn, movie_info AS mi WHERE cn.country_code = '[us]' AND mi.note LIKE '%internet%' AND t.production_year > 1990 AND t.id = at.movie_id AND at.movie_id = t.id AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = at.movie_id AND at.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = at.movie_id AND at.movie_id = mi.movie_id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;