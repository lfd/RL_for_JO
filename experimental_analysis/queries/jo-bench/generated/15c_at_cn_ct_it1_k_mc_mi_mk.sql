SELECT * FROM info_type AS it1, movie_info AS mi, movie_companies AS mc, company_type AS ct, movie_keyword AS mk, keyword AS k, company_name AS cn, aka_title AS at WHERE cn.country_code = '[us]' AND it1.info = 'release dates' AND mi.note LIKE '%internet%' AND mi.info IS NOT NULL AND (mi.info LIKE 'USA:% 199%' OR mi.info LIKE 'USA:% 200%') AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = at.movie_id AND at.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = at.movie_id AND at.movie_id = mi.movie_id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;